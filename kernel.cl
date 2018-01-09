
#define MAX_SIZE 4000

// вычисляется по строке за раз
// строка первой входной матрицы кешируется
// вторая входная матрица хрнится по столбцам
__kernel void main(__global const float* in1,
                   __global const float* in2,
                   __global float* out,
                   const int r1_size,
                   const int c1_r2_size,
                   const int c2_size)
{
    const int row = get_global_id(0);

    // буфер строки матрицы in1
    float row_buf[MAX_SIZE];
    for(int col = 0; col < c1_r2_size; col++)
        row_buf[col] = in1[row * c1_r2_size + col];

    __local float col_buf[MAX_SIZE]; // буфер столбца матрицы in2
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);

    for(int col = 0; col < c2_size; col++) {
        const int part_end = min(local_size * (local_id + 1), c1_r2_size);
        for(int row2 = local_id * local_size; row2 < part_end; row2++)
            col_buf[row2] = in2[col * c1_r2_size + row2];

        barrier(CLK_LOCAL_MEM_FENCE);

        float sum = 0.0f;
        for(int col_row = 0; col_row < c1_r2_size; col_row++) {
            //sum += in1[row * c1_r2_size + col_row] * in2[col * c1_r2_size + col_row];
            //sum += row_buf[col_row] * in2[col * c1_r2_size + col_row];
            sum += row_buf[col_row] * col_buf[col_row];
        }
        out[row * c2_size + col] = sum;
    }
}
