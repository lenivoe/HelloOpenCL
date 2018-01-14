
#define MAX_SIZE 8000

// вычисляется по строке за раз
// вторая входная матрица хрнится по столбцам
// строка первой входной матрицы кешируется в приватную память
// стобец второй входной матрицы кешируется в локальную память группы
// основная масса вычислений использует буферы как массивы четырехмерных векторов
__kernel void main(__global const float* in1,   // первая входная матрица
                   __global const float* in2,   // вторая входная матрица
                   __global float* out,         // выходная матрица
                   const int r1_count,          // количество строк первой матрицы
                   const int c1_r2_count,       // количество столбцов первой матрицы и одновременно строк второй
                   const int c2_count,          // количество столбцов второй матрицы
                   __local float* col_buf)      // буфер стобца второй матрицы
{
    const int row = get_global_id(0);
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);

    // инициализация буфера строки матрицы in1 векторами по 4 элемента
    float row_buf[MAX_SIZE];
    for(int col = 0; col < c1_r2_count; col++)
        row_buf[col] = in1[row * c1_r2_count + col];

    // переменные для доступа к буферам через вектора
    const int vec_buf_size = c1_r2_count / 4;
    const float4* row_vec_buf = row_buf;
    __local const float4* col_vec_buf = col_buf;

    // вычисление строки выходной матрицы
    for(int col = 0; col < c2_count; col++) {
        // Заполение буфера столбца, локального для группы.
        // Каждый рабочий элемент группы выполняет заполнение своей части,
        // размер которой равен local_size, кроме последней части столбца, размер которой равен c1_r2_count % local_size.
        const int part_end = min(local_size * (local_id + 1), c1_r2_count);
        for(int row2 = local_id * local_size; row2 < part_end; row2++)
            col_buf[row2] = in2[col * c1_r2_count + row2];

        barrier(CLK_LOCAL_MEM_FENCE); // ожидание конца инициализации всех частей буфера столбца

        // вычисление элемента выходной матрицы
        float sum = 0.0f;
        for(int col_row = 0; col_row < vec_buf_size; col_row++)
            sum += dot(row_vec_buf[col_row], col_vec_buf[col_row]);

        const int buf_part_end = vec_buf_size * 4;
        for(int col_row = buf_part_end; col_row < c1_r2_count; col_row++)
            sum += row_buf[col_row] * col_buf[col_row];

        out[row * c2_count + col] = sum;
    }
}
