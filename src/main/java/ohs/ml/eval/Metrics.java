package ohs.ml.eval;

public class Metrics {

	public static double f1(double precision, double recall) {
		double ret = 0;
		if (precision > 0 && recall > 0) {
			ret = 2 * (precision * recall) / (precision + recall);
		}
		return ret;
	}

	public static double f1(int correct_cnt, int answer_cnt, int predict_cnt) {
		return f1(precision(correct_cnt, predict_cnt), recall(correct_cnt, answer_cnt));
	}

	public static double precision(int correct_cnt, int predict_cnt) {
		double ret = 0;
		if (predict_cnt > 0) {
			ret = 1f * correct_cnt / predict_cnt;
		}
		return ret;
	}

	public static double recall(int correct_cnt, int answer_cnt) {
		double ret = 0;
		if (answer_cnt > 0) {
			ret = 1f * correct_cnt / answer_cnt;
		}
		return ret;
	}

}
