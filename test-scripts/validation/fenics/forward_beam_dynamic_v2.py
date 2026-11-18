from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
import os
from tet10_mesh_utils import load_tetgen_mesh_from_files, locate_raw_node_dof
import time
from matplotlib import pyplot as plt 
import argparse
import csv

# Versions
# v1: Initial solver
# v2: Replaced LinearProblem.solve with PETsc matricies
#     Created class to study timing
#
# Next Steps: 
#   Lumped mass?

class ForwardDynamic():

    def __init__(self, comm, length=3, height=2, width=1, load=10000, E=7e8, nu=0.33, gdim=3, res=0, steps=5001,
                 t_final=50e-3, debug=False, write_output=False):
        # Config
        self.DEBUG = debug
        self.write_output= write_output
        self.comm = comm
        self.L = length         # Length of beam (x) 
        self.H = height         # Height of beam (y)
        self.W = width          # Width of beam (z)
        self.load = load        # Applied load
        self.E = E              # Young's modulus
        self.nu = nu            # Poisson ratio
        self.gdim = gdim        # Dimension of mesh
        self.RES = res          # resolution of the mesh
        self.STEPS = steps      # number of solver steps
        self.t_final = t_final  # simulation end time

        self.rank = MPI.COMM_WORLD.rank
        self.f_ext = None
        self.bc = None
        self.P = None
        self.M = None
        self.boundary_dof = None

        # Find the mesh files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
        mesh_dir = os.path.join(self.project_root, "data", "meshes", "T10", "resolution")
        node_file = os.path.join(mesh_dir, f"beam_3x2x1_res{self.RES}.1.node")
        ele_file = os.path.join(mesh_dir, f"beam_3x2x1_res{self.RES}.1.ele")

        # Load TetGen mesh
        self.domain, _ = load_tetgen_mesh_from_files(node_file, ele_file)
        # Create function space
        self.V = fem.functionspace(self.domain, ("Lagrange", 2, (self.domain.geometry.dim, )))
        dofmap = self.V.dofmap
        self.total_dofs = dofmap.index_map.size_global * dofmap.index_map_bs

        # Create functions
        # In gneral need test function (v) and unknown function (u)
        # both are part of function space V
        # The solution, u exists in V, so this object is where the solution to the BVP exists
        # u is a space of possible solutions
        self.u = fem.Function(self.V)
        self.v = ufl.TestFunction(self.V)
        self.u.name = "Displacement"

        # Tracked Node
        dof_coords = self.V.tabulate_dof_coordinates()
        num_owned_dofs = dofmap.index_map.size_local
        self.tracked_node_position, self.tracked_node_dof, self.tracked_node_coord, tracked_node_rank = locate_raw_node_dof(
            self.domain, self.rank, dof_coords, num_owned_dofs, node_file, 24
        )

        if self.rank == 0 and self.DEBUG:
            print(
                f"Tracked node raw id {self.tracked_node_dof}: "
                f"({self.tracked_node_position[0]:.6f}, {self.tracked_node_position[1]:.6f}, "
                f"{self.tracked_node_position[2]:.6f})"
            )
            print(f"  Owned by rank: {tracked_node_rank}")
        if self.tracked_node_dof is not None and self.DEBUG:
            print(f"  DOF node index: {self.tracked_node_dof}")
            print(
                f"  Actual coordinates: ({self.tracked_node_coord[0]:.6f}, "
                f"{self.tracked_node_coord[1]:.6f}, {self.tracked_node_coord[2]:.6f})"
            )

    def setupBCs(self):
        # fix x = 0 face
        def on_boundary(x):
            return np.isclose(x[0], 0)

        self.boundary_dof = fem.locate_dofs_geometrical(self.V, on_boundary)
        u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
        self.bc = fem.dirichletbc(u_zero, self.boundary_dof, self.V)

    def setupLoads(self):
        dof_coords = self.V.tabulate_dof_coordinates() # Indicies 0 - num_owned_dofs -1 are owned by current process
        dofmap = self.V.dofmap
        num_owned_dofs = dofmap.index_map.size_local
        block_size = dofmap.index_map_bs
        force_dofs = []

        for i, coord in enumerate(dof_coords):
            if i < num_owned_dofs and abs(coord[0]-self.L) < 1e-6:
                force_dofs.append(i)

        # Calculate force per node
        local_num_force_nodes = len(force_dofs)
        global_num_force_nodes = self.domain.comm.allreduce(local_num_force_nodes, op=MPI.SUM)

        force_per_node = self.load / global_num_force_nodes if global_num_force_nodes > 0 else 0.0

        if self.rank == 0 and self.DEBUG:
            print(f"Applying Lumped Force (+z): {force_per_node} N per node on {global_num_force_nodes} nodes.")
            print(f"  Force will be applied for 1 second, then turned off.")

        # Create the vector
        f_temp = fem.Function(self.V)
        f_temp.x.array[:] = 0.0
        for node in force_dofs:
            # assign forces at each node using k indexing
            f_temp.x.array[node * block_size + 2] = force_per_node

        # Move to PETSc vector
        self.f_ext = f_temp.x.petsc_vec.copy()
        self.f_ext.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def setupMaterialModel(self):
        # St. Venant-Kirchhoff
        E_val = default_scalar_type(self.E)
        nu_val = default_scalar_type(self.nu)

        # Kinematics
        I = ufl.Identity(self.gdim)

        # Form the right Cauchy-Green Tensor
        F = ufl.variable(I+ufl.grad(self.u)) # deformation gradient
        C = F.T * F

        # Lame Parameters
        lambda_svk = fem.Constant(self.domain, E_val * nu_val / ((1+nu_val)*(1-2*nu_val)))
        mu_svk = fem.Constant(self.domain, E_val / (2 * (1 + nu_val)))

        self.P = lambda_svk * ( 0.5 * ufl.tr(C) - 1.5) * F + mu_svk * F * (C - I)
        
    
    def assembleMassMatrix(self):
        metadata = {"quadrature_degree": 5}
        dx = ufl.Measure("dx", domain=self.domain, metadata=metadata)
        rho_val = fem.Constant(self.domain, 2700.0) # TODO add as input

        mass_trial = ufl.TrialFunction(self.V)
        mass_test = ufl.TestFunction(self.V)
        M_form = fem.form(rho_val * ufl.inner(mass_trial, mass_test) * dx)
        if self.bc is None:
            if self.rank == 0: print("Form BC matrix first")
            return
        self.M = assemble_matrix(M_form, bcs=[self.bc])
        self.M.assemble()
        # Set the ksp parameters
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(self.M)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        self.ksp.setUp()

    def solve(self):
        # Set up time integration
        dt = self.t_final / (self.STEPS-1)
        num_to_write = 50

        # Reuse the configured preonly/LU/MUMPS solver from assembleMassMatrix().
        # This keeps KSP and PC settings consistent for every resolution.
        ksp = self.ksp

        # Forward Euler Variational Form
        # This is the sum of external, body, and internal forces
        metadata = {"quadrature_degree": 5}
        dx = ufl.Measure("dx", domain=self.domain, metadata=metadata)
        f_int_form = fem.form(ufl.inner(ufl.grad(self.v), self.P) * dx)

        # Initialize previous u and v states
        u_prev = fem.Function(self.V)
        v_prev = fem.Function(self.V)
        u_prev.x.array[:] = 0.0
        u_prev.x.scatter_forward()
        v_prev.x.array[:] = 0.0
        v_prev.x.scatter_forward()

        # Create the acceleration
        accel = fem.Function(self.V)
        accel.x.array[:] = 0.0
        accel.x.scatter_forward()

        node_history = [] # Tracked node positions

        if self.rank == 0 and self.DEBUG:
            print(f"\nStarting dynamic analysis with dt={dt:.2e}\n")

        # Create work vectors for explicit time stepping
        # We copy f_ext because it is already allocated and split among ranks. 
        residual = self.f_ext.copy()
        f_int = self.f_ext.copy()

        force_off_step = self.STEPS // 2 # match c++ implementation

        # Start Dynamic Analysis
        # Measure the time-stepping loop only. This includes Python, PETSc/MPI,
        # and tracked-node gathering overhead, but excludes setup and CSV output.
        self.comm.Barrier()
        time_start = time.perf_counter()
        for i in range(self.STEPS):
            t = i * dt
            # Turn off force after first half of steps
            if i == force_off_step:
                self.f_ext.zeroEntries()
                self.f_ext.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                if self.rank == 0:
                    print(f"Force turned off at step {i} (t = {t:.4f}s)")

            self.u.x.array[:] = u_prev.x.array[:]
            self.u.x.scatter_forward()

            # Assign owned internal force vector to 0.
            with f_int.localForm() as f_int_local:
                f_int_local.set(0.0)

            # Assemble internal force vector using preallocated memory at f_int
            assemble_vector(f_int, f_int_form)
            f_int.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

            # Set the boundary conditions
            fem.petsc.set_bc(f_int, [self.bc])

            # Compute residual
            residual.zeroEntries()
            residual.axpy(1.0, self.f_ext)
            residual.axpy(-1.0, f_int)

            # Solve the problem at the timestep
            ksp.solve(residual, accel.x.petsc_vec)
            accel.x.scatter_forward() # update ghost nodes

            # Update velocity and displacement using acceleration
            v_next = v_prev.x.array[:] + dt * accel.x.array[:]

            # block size
            block_size = self.V.dofmap.index_map_bs
            
            # Enforce boundary conditions
            for dof in self.boundary_dof:
                for c in range(block_size):
                    k = dof*block_size + c
                    if k < len(v_next):
                        v_next[k] = 0.0

            # Update displacements
            u_next = u_prev.x.array[:] + dt * v_next
            
            self.u.x.array[:] = u_next
            self.u.x.scatter_forward()

            local_position = None

            if self.tracked_node_dof is not None:
                # Get displacement at tracked node (all components)
                u_x_at_node = self.u.x.array[self.tracked_node_dof * block_size + 0]
                u_y_at_node = self.u.x.array[self.tracked_node_dof * block_size + 1]
                u_z_at_node = self.u.x.array[self.tracked_node_dof * block_size + 2]
                x_position = self.tracked_node_coord[0] + u_x_at_node
                y_position = self.tracked_node_coord[1] + u_y_at_node
                z_position = self.tracked_node_coord[2] + u_z_at_node
                local_position = [float(x_position), float(y_position), float(z_position)]
                
            # Gather tracked node data from all ranks to rank 0
            all_positions = self.domain.comm.gather(local_position, root=0)
            if self.rank == 0:
                # Find the non-None position (from the rank that owns the node)
                node_position = next((pos for pos in all_positions if pos is not None), None)
                node_history.append(node_position)

            # Update previous values for next time step
            u_prev.x.array[:] = u_next
            u_prev.x.scatter_forward()

            v_prev.x.array[:] = v_next
            v_prev.x.scatter_forward()

        self.comm.Barrier()
        local_time_stepping_seconds = time.perf_counter() - time_start
        self.time_stepping_seconds = self.comm.allreduce(
            local_time_stepping_seconds, op=MPI.MAX
        )

        root_output_dir = os.path.join(self.project_root, "output")
        if self.rank == 0 and len(node_history) > 0 and self.write_output:
            os.makedirs(root_output_dir, exist_ok=True)

            final_time_ms = self.t_final * 1e3
            csv_filename = (
                f"forward_beam_res{self.RES}_steps{self.STEPS}_"
                f"t{final_time_ms:g}ms_load{self.load:g}N.csv"
            )
            csv_path = os.path.join(root_output_dir, csv_filename)
            
            with open(csv_path, 'w') as f:
                f.write(f"# raw_node_id: {24}\n")
                f.write("step,x_position,y_position,z_position\n")
                for i, (x_val, y_val, z_val) in enumerate(node_history):
                    f.write(f"{i},{x_val:.17f},{y_val:.17f},{z_val:.17f}\n")
            
            print(f"Wrote tracked node x,y,z position history to {csv_path}")
            print(f"  Node position: ({self.tracked_node_position[0]:.1f}, {self.tracked_node_position[1]:.1f}, {self.tracked_node_position[2]:.1f})")
            print(f"  Total steps: {len(node_history)}\n")

        if self.rank == 0 and len(node_history) > 0:
            print(f"Final position, Resolution = {self.RES}: ")
            print(f"  x: {node_history[-1][0]:.17f}")
            print(f"  y: {node_history[-1][1]:.17f}")
            print(f"  z: {node_history[-1][2]:.17f}")

        # Clean up PETSc objects
        residual.destroy()
        f_int.destroy()
        self.f_ext.destroy()
        self.M.destroy()
        self.ksp.destroy()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the explicit beam dynamics simulation."
    )
    parser.add_argument(
        "resolution",
        nargs="?",
        default="0",
        choices=("0", "2", "4", "8", "16", "32", "all"),
        help="mesh resolution to run, or 'all' (default: 0)",
    )
    parser.add_argument(
        "write_output",
        nargs="?",
        default="false",
        type=str.lower,
        choices=("true", "false"),
        help="write tracked node history CSV (default: false)",
    )
    parser.add_argument(
        "steps",
        nargs="?",
        default="5001",
        help="number of simulation steps",
    )
    args = parser.parse_args()

    run_all_resolutions = args.resolution == "all"
    resolutions = [0, 2, 4, 8, 16, 32] if run_all_resolutions else [int(args.resolution)]
    write_output = args.write_output == "true"
    steps = int(args.steps)

    comm = MPI.COMM_WORLD
    study_rows = []

    for res in resolutions:
        sim = ForwardDynamic(comm, write_output=write_output, res=res, steps=steps)
        sim.setupBCs()
        sim.setupLoads()
        sim.setupMaterialModel()
        sim.assembleMassMatrix()
        sim.solve()
        time_elapsed = sim.time_stepping_seconds
        if MPI.COMM_WORLD.rank == 0:
            print(f"Time-stepping loop wall time, Resolution = {res}:")
            print(f"  {time_elapsed:.4e}s")
            print(f"  Average per step: {time_elapsed / steps:.4e}s")
            print(f"  Total vector DOFs: {sim.total_dofs}")

        study_rows.append((res, sim.total_dofs, time_elapsed, sim.t_final))
        del sim

    if MPI.COMM_WORLD.rank == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
        root_output_dir = os.path.join(project_root, "output")
        os.makedirs(root_output_dir, exist_ok=True)
        csv_path = os.path.join(root_output_dir, f"fenics_rtf_pairs_{steps}_steps.csv")
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "resolution", "dof_count", "real_time_factor",
                "solve_time_seconds", "simulation_time_seconds", "steps",
            ])
            for resolution, dof_count, solve_time, simulation_time in study_rows:
                writer.writerow([
                    resolution, dof_count, solve_time / simulation_time,
                    solve_time, simulation_time, steps,
                ])
        print(f"Wrote resolution study CSV: {csv_path}")

    plot = False

    if MPI.COMM_WORLD.rank == 0 and run_all_resolutions and plot:
        plt.plot(np.log(np.asarray(resolutions) + 1), np.log([r[2] for r in study_rows]))
        plt.xlabel("log(Resolution + 1)")
        plt.ylabel("log(Time (s))")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
        root_output_dir = os.path.join(project_root, "output")
        os.makedirs(root_output_dir, exist_ok=True)
        plt.savefig(os.path.join(root_output_dir, "time_plot_log"))
