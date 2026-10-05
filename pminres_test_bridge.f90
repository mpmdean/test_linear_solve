! Test-only C interface to the unchanged production PMINRES and CSR matvec.
module pminres_test_bridge
    use iso_c_binding
    use m_constants, only: dp, czero
    use m_control
    use m_types
    use mpi
    implicit none
    type(T_csr), allocatable :: blocks(:)
    integer, allocatable :: bounds(:,:,:), needed(:,:)
    integer :: local_size
contains
    subroutine setup(n, nnz, rowptr, columns, values, offsets, comm, tol, maxits) bind(C)
        integer(c_int), value :: n, nnz, comm, maxits
        integer(c_int), intent(in) :: rowptr(n+1), columns(nnz), offsets(*)
        complex(c_double_complex), intent(in) :: values(nnz)
        real(c_double), value :: tol
        integer :: ierr, owner, row, j, k, count_nnz
        integer, allocatable :: local_needed(:), gathered(:,:)
        new_comm = comm
        call MPI_COMM_RANK(new_comm, myid, ierr)
        call MPI_COMM_SIZE(new_comm, nprocs, ierr)
        master = 0
        local_size = n
        linsys_tol = tol
        linsys_max = maxits
        allocate(blocks(nprocs), bounds(2,2,nprocs), needed(nprocs,nprocs))
        allocate(local_needed(nprocs), gathered(nprocs,nprocs))
        local_needed = 0
        do owner=1,nprocs
            bounds(1,:,owner) = offsets(owner) + 1
            bounds(2,:,owner) = offsets(owner+1)
            count_nnz = count(columns >= offsets(owner) .and. columns < offsets(owner+1))
            call init_csr(blocks(owner))
            call alloc_csr(n, count_nnz, blocks(owner))
            blocks(owner)%m = n
            blocks(owner)%nnz = count_nnz
            blocks(owner)%row_shift = offsets(myid+1)
            blocks(owner)%col_shift = offsets(owner)
            if (count_nnz > 0 .and. owner /= myid+1) local_needed(owner) = 1
            k = 1
            do row=1,n
                blocks(owner)%iaa(row) = k
                do j=rowptr(row)+1,rowptr(row+1)
                    if (columns(j) < offsets(owner) .or. columns(j) >= offsets(owner+1)) cycle
                    blocks(owner)%jaa(k) = columns(j) + 1
                    blocks(owner)%aa(k) = values(j)
                    k = k + 1
                enddo
            enddo
            blocks(owner)%iaa(n+1) = k
        enddo
        call MPI_ALLGATHER(local_needed, nprocs, MPI_INTEGER, gathered, nprocs, MPI_INTEGER, new_comm, ierr)
        needed = transpose(gathered)
    end subroutine

    subroutine solve(b, x) bind(C)
        complex(c_double_complex), intent(in) :: b(local_size)
        complex(c_double_complex), intent(out) :: x(local_size)
        integer :: info
        call pminres_csr(nprocs, bounds, needed, local_size, blocks, b, x, info)
        ! Production info is always zero; the Python test checks the true residual.
        flush(6)
    end subroutine

    subroutine multiply(x, y) bind(C)
        complex(c_double_complex), intent(in) :: x(local_size)
        complex(c_double_complex), intent(out) :: y(local_size)
        y = czero
        call pspmv_csr(new_comm, nprocs, bounds, needed, local_size, local_size, blocks, x, y)
    end subroutine

    subroutine cleanup() bind(C)
        integer :: owner
        do owner=1,nprocs
            call dealloc_csr(blocks(owner))
        enddo
        deallocate(blocks, bounds, needed)
    end subroutine
end module
