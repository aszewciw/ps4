
/*
  Author: Mohammed A. Al Farhan, PhD student at ECRC, KAUST
  Email:  mohammed.farhan@kaust.edu.sa

  ps4.c: A PETSc example that solves a 2D linear Poisson equation

  The example code is based upon:
    1) "PETSC FOR PARTIAL DIFFERENTIAL EQUATIONS" book by
        Professor ED BUELER:
        https://github.com/bueler/p4pdes/releases
    2) PETSc KSP Example 50 (ex50):
        $PETSC_DIR/src/ksp/ksp/examples/tutorials/ex50.c
    3) PETSc KSP Example 15 (ex15):
        $PETSC_DIR/src/ksp/ksp/examples/tutorials/ex15.c
*/

#include <stdio.h>
#include <petscdmda.h>
#include <petscksp.h>

/*
  User-functions prototypes
*/
PetscErrorCode
compute_rhs(KSP, Vec, void *);
PetscErrorCode
compute_opt(KSP, Mat, Mat, void *);
PetscErrorCode
test_convergence_rate(KSP, Vec);

/*
  Main C function
*/
int
main(int argc, char * args[])
{
  PetscErrorCode ierr; // PETSc error code

  // Initialize the PETSc database and MPI
  // "petsc.opt" is the PETSc database file
  ierr = PetscInitialize(&argc, &args, NULL, NULL);
  CHKERRQ(ierr); // PETSc error handler

  // The global PETSc MPI communicator
  MPI_Comm comm = PETSC_COMM_WORLD;
  /*
    PETSc DM
    Create a default 16x16 2D grid object
    The minus sign means that the grid x and y dimensions
    are changeable through the command-line options
  */
  DM da;

  ierr = DMDACreate1d(comm, DM_BOUNDARY_GHOSTED, -17, 1, 1, NULL, &da);

  CHKERRQ(ierr);

  /*
    PETSc Vec
  */
  /*  Vector of unknowns approximating \varphi_{i,j}
      on the grid */
  Vec u;
  // Create a global vector derived from the DM object
  // "Global" means "distributed" in MPI language
  ierr = DMCreateGlobalVector(da, &u);
  CHKERRQ(ierr);

  /*
    The right-hand side vector approximating the values of f_{i,j}
  */
  Vec b;
  // Duplicate creates a new vector of the same type as u
  ierr = VecDuplicate(u, &b);
  CHKERRQ(ierr);

  /*
    Krylov Subspace (KSP) object to solve the linear system
  */
  KSP ksp;
  // Create the KPS object
  ierr = KSPCreate(comm, &ksp);
  CHKERRQ(ierr);
  // Set the DM to be used as preconditioner
  ierr = KSPSetDM(ksp, (DM) da);
  CHKERRQ(ierr);
  // Compute the right-hand side vector b
  ierr = KSPSetComputeRHS(ksp, compute_rhs, NULL);
  CHKERRQ(ierr);
  // Compute and assemble the coefficient matrix A
  ierr = KSPSetComputeOperators(ksp, compute_opt, NULL);
  CHKERRQ(ierr);

  // KSP options can be changed during the runtime
  ierr = KSPSetFromOptions(ksp);
  CHKERRQ(ierr);

  // Solve the linear system using KSP
  ierr = KSPSolve(ksp, b, u);
  CHKERRQ(ierr);

  // Verifies the implementation by comparing the
  // numerical solution to the analytical solution
  // The function computes a norm of the difference
  // between the computed solution and the exact solution
  ierr = test_convergence_rate(ksp, u);
  CHKERRQ(ierr);

  /*
    Cleanup the allocations, and exit
  */
  ierr = VecDestroy(&u);
  CHKERRQ(ierr);
  ierr = VecDestroy(&b);
  CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);
  CHKERRQ(ierr);
  ierr = DMDestroy(&da);
  CHKERRQ(ierr);

  // Exit the MPI communicator and finalize the PETSc
  // initialization objects
  ierr = PetscFinalize();
  CHKERRQ(ierr);

  return 0;
}

/*
  Compute the right-hand side vector
*/
PetscErrorCode
compute_rhs(KSP ksp, Vec b, void * ctx)
{
  PetscFunctionBegin;

  PetscErrorCode ierr;

  /* Get the DM oject of the KSP */
  DM da;
  ierr = KSPGetDM(ksp, &da);
  CHKERRQ(ierr);

  /* Get the global information of the DM grid*/
  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);
  CHKERRQ(ierr);

  /* Grid spacing */
  double hx = (1.f / (double) (grid.mx - 1));
  // double hy = (1.f / (double) (grid.my - 1));

  /*  A pointer to access b, the right-hand side PETSc vector
      viewed as a C array */
  double * _b;
  ierr = DMDAVecGetArray(da, b, &_b);
  CHKERRQ(ierr);

  unsigned int i;
  for(i = grid.xs; i<(grid.xs + grid.xm); i++)
  {
    double x = i * hx;

    /* Nodes on the boundary layers (\Gamma) */
    if((i==0) || (i==(grid.mx - 1)))
    {
      _b[i] = 0.0;
    }
    else
    {
      /* Interior nodes in the domain (\Omega) */
      double x2 = x * x;
      double f = hx * hx * (12.0 * x2 - 2.0);
      _b[i] = f;
    }

  }

  // Release the resource
  ierr = DMDAVecRestoreArray(da, b, &_b);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Compute the operator matrix A */
PetscErrorCode
compute_opt(KSP ksp, Mat A, Mat J, void * ctx)
{
  PetscFunctionBegin;

  PetscErrorCode ierr;

  // Get the DMDA object
  DM da;
  ierr = KSPGetDM(ksp, &da);
  CHKERRQ(ierr);

  // Get the grid information
  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);
  CHKERRQ(ierr);

  /*  A PETSc data structure to store information
      about a single row or column in the stencil */
  MatStencil idxm;
  MatStencil idxn[3];

  // The matrix values
  double v[3];

  double hx = (1.f / (double) (grid.mx - 1));
  // double hy = (1.f / (double) (grid.my - 1));

  /* Loop over the grid points */
  unsigned int i;
  for(i = grid.xs; i < (grid.xs + grid.xm); i++)
  {

    idxm.i = i;
    idxn[0].i = i;
    size_t n = 1;

    if ((i==0) || (i==grid.xm - 1) )
    {
      v[0] = 1.f;
    }
    else
    {
      v[0] = 2.f;

      if((i - 1) > 0)
      {
        // Get the column indices
        idxn[n].i = i - 1;
        v[n] = - 1.0;

        n++; // One column added
      }
      if((i + 1) < (grid.mx - 1))
      {
        // Get the column indices
        idxn[n].i = i + 1;
        v[n] = - 1.0;

        n++; // One column added
      }
    }
    ierr = MatSetValuesStencil(A, 1, (const MatStencil *) &idxm,
                              (PetscInt) n, (const MatStencil *) &idxn,
                              (PetscScalar *) v, INSERT_VALUES);
    CHKERRQ(ierr);
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A ,MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Test the convergence rate */
PetscErrorCode
test_convergence_rate(KSP ksp, Vec u)
{
  PetscFunctionBegin;

  PetscErrorCode ierr;

  // Get the DMDA object
  DM da;
  ierr = KSPGetDM(ksp, &da);
  CHKERRQ(ierr);

  // Get the grid information
  DMDALocalInfo grid;
  ierr = DMDAGetLocalInfo(da, &grid);
  CHKERRQ(ierr);

  // Create a global vector
  Vec u_;
  ierr = VecDuplicate(u, &u_);
  CHKERRQ(ierr);

  double * _u;

  double hx = (1.f / (double) (grid.mx - 1));

  // Get a pointer to the PETSc vector
  ierr = DMDAVecGetArray(da, u_, &_u);
  CHKERRQ(ierr);

  unsigned int i;
  for(i = grid.xs; i < (grid.xs + grid.xm); i++){
    double x = i * hx;
    double x2 = x * x;
    double x4 = x2 * x2;
    _u[i] = (x2 - x4);
  }

  ierr = DMDAVecRestoreArray(da, u_, &_u);
  CHKERRQ(ierr);

  double val = 0.f;

  ierr = VecAXPY(u, -1.f, u_);
  CHKERRQ(ierr);
  ierr = VecNorm(u, NORM_INFINITY, (PetscScalar *) &val);
  CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,
                    "Numerical Error [NORM_INFINITY]: \t %g\n", val);
  CHKERRQ(ierr);

  ierr = VecDestroy(&u_); CHKERRQ(ierr);

  return 0;
}
