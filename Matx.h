#ifndef MATX_H
#define MATX_H

#include <memory>

template<bool BY_ROWS = true>
class CMatx {
public:
    CMatx(size_t _rows = 1, size_t _cols = 1) :
        m_Matx(std::make_unique<float[]>(_rows * _cols)),
        m_Rows(_rows), m_Cols(_cols) { }

    size_t Rows() const { return m_Rows; }
    size_t Cols() const { return m_Cols; }

    float& At(size_t _row, size_t _col);

private:
    std::unique_ptr<float[]> m_Matx;
    size_t m_Rows, m_Cols;
};

template<>
inline float& CMatx<true>::At(size_t _row, size_t _col) { return m_Matx[_row * m_Cols + _col]; }

template<>
inline float& CMatx<false>::At(size_t _row, size_t _col) { return m_Matx[_col * m_Rows + _row]; }

#endif // MATX_H
