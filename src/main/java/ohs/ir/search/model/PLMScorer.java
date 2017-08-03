package ohs.ir.search.model;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ohs.ir.search.app.DocumentSearcher;
import ohs.ir.weight.TermWeighting;
import ohs.matrix.SparseVector;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;

public class PLMScorer extends LMScorer {

	public static final double pi = Math.PI;

	// Cumulative distribution function of Arc Kernel
	public static double ArcCDF(double x, double mean, double sigma) {
		double res;
		x = (x - mean) / sigma;

		if (x >= 1) {
			res = (pi - 1.0) * sigma / 2.0;
		} else if (x < -1) {
			res = 0;
		} else if (x < 0) {
			res = sigma * (Math.asin(x) + pi / 2.0 - Math.sqrt(1 - x * x) + (1 - Math.abs(x)) * (1 - Math.abs(x)) / 2.0)
					/ 2.0;
		} else {
			res = (pi - 1.0) * sigma / 2.0 - sigma
					* (Math.asin(-x) + pi / 2.0 - Math.sqrt(1 - x * x) + (1 - Math.abs(x)) * (1 - Math.abs(x)) / 2.0)
					/ 2.0;
		}
		return res;
	}

	// Cumulative distribution function of Circle Kernel
	public static double CircleCDF(double x, double mean, double sigma) {
		double res;
		x = (x - mean) / sigma;

		if (x >= 1) {
			res = (pi - 2.0) * sigma;
		} else if (x < -1) {
			res = 0;
		} else if (x < 0) {
			res = sigma * (Math.asin(x) + pi / 2.0 - Math.sqrt(1 - x * x));
		} else {
			res = (pi - 2.0) * sigma - sigma * (Math.asin(-x) + pi / 2.0 - Math.sqrt(1 - x * x));
		}
		return res;
	}

	// Cumulative distribution function of Cosine Kernel
	public static double CosineCDF(double x, double mean, double sigma) {
		double res;
		x = (x - mean) / sigma;
		if (x >= 1) {
			res = sigma;
		} else if (x < -1) {
			res = 0;
		} else if (x < 0) {
			res = sigma * (1 + x - Math.sin(pi * x) / pi) / 2.0;
		} else {
			res = sigma - sigma * (1 - x + Math.sin(pi * x) / pi) / 2.0;
		}
		return res;
	}

	public static double GaussianCDF(double x, double mean, double sigma) {
		double res;
		x = (x - mean) / sigma;
		if (x == 0) {
			res = 0.5;
		} else {
			double oor2pi = 1 / (Math.sqrt(2 * pi));
			double t = 1 / (1 + 0.2316419 * Math.abs(x));
			t *= oor2pi * Math.exp(-0.5 * x * x)
					* (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
			if (x >= 0) {
				res = 1 - t;
			} else {
				res = t;
			}
		}
		return res;
	}

	/**
	 * The propagated count from one position to another, when the distance between
	 * two positions is "dis" Propagation function: -1 Passage; 0 Gaussian; 1:
	 * Cosine; 2: Triangle; 3: Arc; 4: Circle Here, "dis" is normalized
	 * 
	 * @param dis
	 * @param propFunction
	 * @return
	 */
	public static double PropagationCount(double dis, int propFunction) {
		dis = Math.abs(dis);
		switch (propFunction) {
		case -1:
			if (dis <= 1.0)
				return 1.0;
			else
				return 0;
		case 1:
			if (dis <= 1.0)
				return (1 + Math.cos(pi * dis)) / 2.0;
			else
				return 0;
		case 2:
			if (dis <= 1.0)
				return 1 - dis;
			else
				return 0;
		case 3:
			if (dis <= 1.0)
				return (1 - dis + Math.sqrt(1 - dis * dis)) / 2.0;
			else
				return 0;
		case 4:
			if (dis <= 1.0)
				return Math.sqrt(1 - dis * dis);
			else
				return 0;
		case 0:
		default:
			return Math.exp(-dis * dis / 2);
		}
	}

	/**
	 * 
	 * The sum of propogated counts from all positions to the current position
	 * "pos", i.e., the size of the vitual passage Propagation function: -1 Passage;
	 * 0 Gaussian; 1: Cosine; 2: Triangle; 3: Arc; 4: Circle
	 * 
	 * @param pos
	 * @param len_d
	 * @param propFunction
	 * @param sigma
	 * @return
	 */
	public static double PropagationCountSum(int pos, int len_d, int propFunction, double sigma) {
		double ret = 0;
		double psg_len = 0;

		switch (propFunction) {
		case -1:
			if (sigma > len_d - pos) {
				psg_len += len_d - pos;
			} else {
				psg_len += sigma;
			}

			if (sigma > pos) {
				psg_len += pos;
			} else {
				psg_len += sigma;
			}
			break;

		case 1:
			psg_len = CosineCDF(len_d, pos, sigma) - CosineCDF(0, pos, sigma);
			break;
		case 2:
			psg_len = TriangleCDF(len_d, pos, sigma) - TriangleCDF(0, pos, sigma);
			break;
		case 3:
			psg_len = ArcCDF(len_d, pos, sigma) - ArcCDF(0, pos, sigma);
			break;
		case 4:
			psg_len = CircleCDF(len_d, pos, sigma) - CircleCDF(0, pos, sigma);
			break;
		case 0:
		default:
			psg_len = Math.sqrt(2 * pi) * sigma * (GaussianCDF(len_d, pos, sigma) - GaussianCDF(0, pos, sigma));
			break;
		}

		return psg_len;
	}

	// Cumulative distribution function of Triangle Kernel
	public static double TriangleCDF(double x, double mean, double sigma) {
		double res;
		x = (x - mean) / sigma;
		if (x >= 1) {
			res = sigma;
		} else if (x < -1) {
			res = 0;
		} else if (x < 0) {
			res = sigma * (1 - Math.abs(x)) * (1 - Math.abs(x)) / 2.0;
		} else {
			res = sigma - sigma * (1 - x) * (1 - x) / 2.0;
		}
		return res;
	}

	private int propFunction = 0;

	private double sigma = 175;

	public PLMScorer(DocumentSearcher ds) {
		super(ds);
	}

	public Map<Integer, Integer> getQueryWordLocations(SparseVector Q, IntegerArray d) {
		Map<Integer, Integer> ret = Generics.newHashMap();
		for (int j = 0; j < d.size(); j++) {
			int w = d.get(j);
			if (Q.location(w) < 0) {
				continue;
			}
			ret.put(j, w);
		}
		return ret;
	}

	@Override
	public SparseVector scoreFromIndex(SparseVector Q, SparseVector docs) throws Exception {
		SparseVector ret = docs.copy();
		ret.setAll(0);

		for (int i = 0; i < docs.size(); i++) {
			int dseq = docs.indexAt(i);
			IntegerArray d = dc.get(dseq).getSecond();
			int len_d = d.size();

			Map<Integer, Integer> locToWords = getQueryWordLocations(Q, d);

			SparseVector plmScores = new SparseVector(locToWords.size());

			List<Integer> locs = Generics.newArrayList(locToWords.keySet());

			for (int j = 0; j < locs.size(); j++) {
				int loc = locs.get(j);
				int center = locToWords.get(loc);
				double len_psg = PropagationCountSum(center, len_d, propFunction, sigma);

				SparseVector plm = Q.copy();
				plm.setAll(0);

				for (Entry<Integer, Integer> e : locToWords.entrySet()) {
					int pos = e.getKey();
					int w = e.getValue();
					double prop_cnt = PropagationCount((pos - center) / sigma, propFunction);
					double pr = prop_cnt / len_psg;
					plm.add(w, pr);
				}

				for (int k = 0; k < plm.size(); k++) {
					int w = plm.indexAt(k);
					double pr_w_in_psg = plm.valueAt(k);
					double pr_w_in_c = vocab.getProb(w);
					double pr_w_in_d_dir = TermWeighting.dirichletSmoothing(pr_w_in_psg, len_psg, pr_w_in_c, prior_dir);
					plm.setAt(k, pr_w_in_d_dir);
				}
				plm.summation();

				double div_sum = 0;

				for (int k = 0; k < Q.size(); k++) {
					int w = Q.indexAt(k);
					double pr_w_in_q = Q.valueAt(k);
					double pr_w_in_d = plm.value(w);
					if (pr_w_in_d > 0) {
						div_sum += pr_w_in_q * Math.log(pr_w_in_q / pr_w_in_d);
					}
				}

				double score = Math.exp(-div_sum);
				plmScores.addAt(j, center, score);
			}
			double score = plmScores.size() == 0 ? 0 : plmScores.max();
			ret.addAt(i, dseq, score);

		}
		return ret;
	}

	public void setPropType(int propFunction) {
		this.propFunction = propFunction;
	}

	public void setSigma(double sigma) {
		this.sigma = sigma;
	}

}
