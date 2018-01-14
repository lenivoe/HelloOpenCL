
// все матрицы хранятся по строкам
__kernel void main(__global const float* _input1,
                   __global const float* _input2,
                   __global float* _output,
                   const int _rows1,
                   const int _cols1_rows2,
                   const int _cols2)
{
    const int row = get_global_id(0) / _cols2;
    const int col = get_global_id(0) % _cols2;

    float sum = 0.0f;
    for(int col_row = 0; col_row < _cols1_rows2; col_row++) {
        sum += _input1[row * _cols1_rows2 + col_row] *
               _input2[col_row * _cols2 + col]; // <- разница тут
    }
    _output[row * _cols2 + col] = sum;
}
