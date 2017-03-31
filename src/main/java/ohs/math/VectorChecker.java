package ohs.math;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.matrix.Matrix;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.matrix.Vector;

public class VectorChecker {

	public static boolean isProductable(Matrix a, Matrix b) {
		return a.colSize() == b.rowSize() ? true : false;
	}

	public static boolean isProductable(Matrix a, Matrix b, Matrix c) {
		int aRowDim = a.rowSize();
		int aColDim = a.colSize();

		int bRowDim = b.rowSize();
		int bColDim = b.colSize();

		int cRowDim = c.rowSize();
		int cColDim = c.colSize();

		if (aRowDim == cRowDim && bColDim == cColDim && aColDim == bRowDim) {
			return true;
		} else {
			return false;
		}
	}

	public boolean isFinite(DenseVector a) {
		return ArrayChecker.isFinite(a.values());
	}

	public boolean isFinite(DenseMatrix a) {
		return ArrayChecker.isFinite(a.values());
	}

	public static boolean isProductable(Matrix a, Vector b) {
		return a.colSize() == b.size() ? true : false;
	}

	public static boolean isProductable(Matrix a, Vector b, Vector c) {
		return isProductable(a, b) && isProductable(b, c) ? true : false;
	}

	public static boolean isProductable(Vector a, Vector b) {
		return a.size() == b.size() ? true : false;
	}

	public static boolean isSameDimension(Vector a, Vector b) {
		return a.size() == b.size() ? true : false;
	}

	public static boolean isSameDimensions(Matrix a, Matrix b) {
		return a.rowSize() == b.rowSize() && a.colSize() == b.colSize() ? true : false;
	}

	public static boolean isSparse(Matrix a) {
		return a instanceof SparseMatrix;
	}

	public static boolean isSparse(Vector a) {
		return a instanceof SparseVector;
	}

}
