package ohs.ir.weight;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import ohs.math.ArrayMath;
import ohs.math.CommonMath;
import ohs.math.VectorMath;
import ohs.matrix.DenseVector;
import ohs.matrix.SparseVector;
import ohs.types.generic.Vocab;

/**
 * This class provides well-known IR utilities.
 * 
 * @author Heung-Seon Oh
 * 
 */
public class TermWeighting {

	public static boolean print_log = false;

	public static final double b = 0.75;

	public static final double k1 = 1.5;

	public static final double k3 = 1000;

	public static final double sigma = 0;

	public static final double prior_dirichlet = 2000;

	public static final double mixture_jm = 2000;

	public static void bm25(Collection<SparseVector> docs) {
		bm25(docs);
	}

	public static void bm25(Collection<SparseVector> docs, DenseVector docFreqs) {
		double len_d_avg = 0;

		for (SparseVector x : docs) {
			len_d_avg += x.sum();
		}

		len_d_avg /= docs.size();

		int w = 0;
		double cnt_w = 0;
		double doc_freq = 0;
		double len_d = 0;
		double num_docs = docs.size();

		for (SparseVector x : docs) {
			len_d = x.sum();
			for (int i = 0; i < x.size(); i++) {
				w = x.indexAt(i);
				cnt_w = x.valueAt(i);
				doc_freq = docFreqs.valueAt(w);
				x.setAt(i, bm25(cnt_w, len_d, len_d_avg, num_docs, doc_freq, b, k1, sigma));
			}
		}
	}

	public static void bm25(Collection<SparseVector> ds, DenseVector docFreqs, double num_docs, double b, double k1, double sigma) {
		double len_d_avg = 0;

		for (SparseVector x : ds) {
			len_d_avg += x.sum();
		}

		len_d_avg /= ds.size();

		for (SparseVector d : ds) {
			bm25(d, docFreqs, num_docs, len_d_avg, b, k1, sigma);
		}
	}

	public static double bm25(double cnt_w_in_d, double len_d, double len_d_avg, double num_docs, double doc_freq, double b, double k1,
			double sigma) {
		double numerator = cnt_w_in_d * (k1 + 1);
		double denominator = cnt_w_in_d + k1 * (1 - b + b * (len_d / len_d_avg));
		// double idf = Math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5));
		double idf = idf(num_docs, doc_freq);
		double ret = idf * (numerator / denominator + sigma);
		return ret;
	}

	/**
	 * 
	 * Lv, Y., & Zhai, C. (2011). Lower-bounding term frequency normalization. In Proceedings of the 20th ACM international conference on
	 * Information and knowledge management - CIKM ’11 (Vol. 51, p. 7). New York, New York, USA: ACM Press.
	 * http://doi.org/10.1145/2063576.2063584
	 * 
	 * @param cnt_w_in_q
	 * @param k3
	 * @param cnt_w_in_d
	 * @param len_d
	 * @param len_d_avg
	 * @param num_docs
	 * @param doc_freq
	 * @param b
	 * @param k1
	 * @param sigma
	 * @return
	 */
	public static double bm25(double cnt_w_in_q, double k3, double cnt_w_in_d, double len_d, double len_d_avg, double num_docs,
			double doc_freq, double b, double k1, double sigma) {
		double ret = bm25(cnt_w_in_d, len_d, len_d_avg, num_docs, doc_freq, b, k1, sigma);
		ret *= ((k3 + 1) * cnt_w_in_q) / (k3 + cnt_w_in_q);
		return ret;
	}

	public static void bm25(List<SparseVector> docs) {
		System.out.println("weight by bm25.");
		double k_1 = 1.2d;
		double k_3 = 8d;
		double b = 0.75d;
		double len_avg_d = 0;

		for (SparseVector x : docs) {
			if (x.size() == 0) {
				continue;
			}
			len_avg_d += x.sum();
		}

		len_avg_d /= docs.size();

		DenseVector term_docFreq = documentFreqs(docs, maxWordIndex(docs) + 1);

		for (int i = 0; i < docs.size(); i++) {
			SparseVector x = docs.get(i);
			double len_d = x.sum();
			double norm = 0;

			for (int j = 0; j < x.size(); j++) {
				int termId = x.indexAt(j);
				double tf = x.valueAt(j);
				double value1 = (tf * (k_1 + 1)) / (tf + k_1 * (1 - b + b * (len_d / len_avg_d)));

				double doc_freq = term_docFreq.value(termId);
				double num_docs = docs.size();
				double value2 = Math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5));
				double weight = value1 * value2;

				x.setAt(j, weight);
				norm += weight * weight;
			}
			x.multiply(1f / norm);
		}
	}

	public static void bm25(SparseVector d, DenseVector docFreqs, double num_docs, double len_d_avg, double b, double k1, double sigma) {
		int w = 0;
		double cnt_w = 0;
		double doc_freq = 0;
		double len_d = d.sum();
		double sum = 0;
		for (int i = 0; i < d.size(); i++) {
			w = d.indexAt(i);
			cnt_w = d.valueAt(i);
			doc_freq = docFreqs.valueAt(w);
			d.setAt(i, bm25(cnt_w, len_d, len_d_avg, num_docs, doc_freq, b, k1, sigma));
			sum += d.valueAt(i);
		}
		d.setSum(sum);
	}

	public static void dfree(List<SparseVector> docs, DenseVector collWordCnts) {
		double num_words = collWordCnts.sum();
		int w = 0;
		double tf = 0;
		double prior = 0;
		double posterior = 0;
		double termFrequency = 0;
		double InvPriorCollection = 0;
		double norm = 0;
		double weight = 0;

		for (int i = 0; i < docs.size(); i++) {
			SparseVector x = docs.get(i);
			double docLength = x.sum();

			for (int j = 0; j < x.size(); j++) {
				w = x.indexAt(j);
				tf = x.valueAt(j);
				prior = tf / docLength;
				posterior = (tf + 1d) / (docLength + 1);
				termFrequency = collWordCnts.value(w);
				InvPriorCollection = num_words / termFrequency;
				norm = tf * CommonMath.log2(posterior / prior);
				weight = norm * (tf * (-CommonMath.log2(prior * InvPriorCollection)) +

						(tf + 1d) * (+CommonMath.log2(posterior * InvPriorCollection)) + 0.5 * CommonMath.log2(posterior / prior));

				x.setAt(j, weight);
			}

			VectorMath.unitVector(x);
		}
	}

	public static double dirichletSmoothing(double pr_w_in_d, double len_d, double pr_w_in_c, double prior_dir) {
		double mixture_dir = prior_dir / (len_d + prior_dir);
		return (1 - mixture_dir) * pr_w_in_d + mixture_dir * pr_w_in_c;
	}

	public static double dirichletSmoothing(int cnt_w_in_d, double len_d, double pr_w_in_c, double prior_dir) {
		double mixture_dir = prior_dir / (len_d + prior_dir);
		double pr_w_in_d = len_d > 0 ? cnt_w_in_d / len_d : 0;
		return (1 - mixture_dir) * pr_w_in_d + mixture_dir * pr_w_in_c;
	}

	public static DenseVector documentFreqs(Collection<SparseVector> docs) {
		return documentFreqs(docs, maxWordIndex(docs) + 1);
	}

	public static DenseVector documentFreqs(Collection<SparseVector> docs, int word_size) {
		DenseVector ret = new DenseVector(word_size);
		for (SparseVector doc : docs) {
			for (int w : doc.indexes()) {
				ret.add(w, 1);
			}
		}
		return ret;
	}

	public static double idf(double num_docs, double doc_freq) {
		return Math.log((num_docs + 1) / (doc_freq));
	}

	public static List<SparseVector> invertedIndexDoubleVector(List<SparseVector> docs, int num_terms) {
		System.out.println("build inverted index.");

		List<Integer[]>[] lists = new List[num_terms];

		for (int i = 0; i < lists.length; i++) {
			lists[i] = new ArrayList<Integer[]>();
		}

		for (int docId = 0; docId < docs.size(); docId++) {
			SparseVector doc = docs.get(docId);
			for (int j = 0; j < doc.size(); j++) {
				int termId = doc.indexAt(j);
				double termCount = doc.valueAt(j);
				lists[termId].add(new Integer[] { docId, (int) termCount });
			}
		}

		List<SparseVector> ret = new ArrayList<SparseVector>();

		for (int termId = 0; termId < lists.length; termId++) {
			List<Integer[]> list = lists[termId];

			int[] docIds = new int[list.size()];
			double[] termCounts = new double[list.size()];

			for (int j = 0; j < list.size(); j++) {
				docIds[j] = list.get(j)[0];
				termCounts[j] = list.get(j)[1];
			}

			SparseVector postingList = new SparseVector(docIds, termCounts);
			postingList.sortIndexes();

			ret.add(postingList);

			lists[termId].clear();
			lists[termId] = null;
		}
		return ret;
	}

	public static double jelinekMercerSmoothing(double pr_w_in_d, double pr_w_in_c, double mixture_jm) {
		return (1 - mixture_jm) * pr_w_in_d + mixture_jm * pr_w_in_c;
	}

	public static int[][] makeInvertedIndex(List<SparseVector> docs, int vocab_size) {
		System.out.println("build inverted index.");

		int[][] ret = new int[vocab_size][];
		for (int i = 0; i < vocab_size; i++) {
			if ((i + 1) % 100 == 0) {
				System.out.printf("\r[%d/%d]", i + 1, vocab_size);
			}
			Set<Integer> set = new TreeSet<Integer>();

			for (int j = 0; j < docs.size(); j++) {
				SparseVector d = docs.get(j);
				if (d.location(i) < 0) {
					continue;
				}
				set.add(j);
			}

			ret[i] = new int[set.size()];
			int loc = 0;
			for (int docId : set) {
				ret[i][loc] = docId;
				loc++;
			}
		}

		System.out.printf("\r[%d/%d]\n", vocab_size, vocab_size);
		return ret;
	}

	public static int maxWordIndex(Collection<SparseVector> docs) {
		int ret = 0;
		for (SparseVector x : docs) {
			if (x.size() == 0) {
				continue;
			}
			for (int idx : x.indexes()) {
				ret = Math.max(idx, ret);
			}
		}
		return ret;
	}

	/**
	 * Hiemstra, D., Robertson, S., and Zaragoza, H. 2004. Parsimonious language models for information retrieval. Proceedings of the 27th
	 * annual international ACM SIGIR conference on Research and development in information retrieval, ACM, 178–185.
	 * 
	 * Na, S.-H., Kang, I.-S., and Lee, J.-H. 2007. Parsimonious translation models for information retrieval. Information Processing &
	 * Management 43, 1, 121–145.
	 * 
	 * @param doc
	 * @param cnts_w_in_c
	 * @param mixture_jm
	 */
	public static void parsimoniousLanguageModel(SparseVector doc, DenseVector cnts_w_in_c, double mixture_jm) {
		int w = 0;
		double cnt_w_in_d = 0;
		double pr_w_in_d = 0;
		double pr_w_in_c = 0;
		double mixture_d = 1 - mixture_jm;
		double e = 0;
		double pr_sum = 0;
		double dist = 0;

		SparseVector dx = doc.copy();
		dx.normalize();

		SparseVector dy = dx.copy();

		for (int j = 0; j < 200; j++) {
			pr_sum = 0;

			for (int k = 0; k < dx.size(); k++) {
				w = dx.indexAt(k);
				pr_w_in_d = dx.valueAt(k);
				cnt_w_in_d = doc.valueAt(k);
				pr_w_in_c = cnts_w_in_c.prob(w);
				e = (mixture_d * pr_w_in_d) / ArrayMath.addAfterMultiply(pr_w_in_c, mixture_jm, pr_w_in_d, mixture_d);
				e = cnt_w_in_d * e;
				pr_sum += e;

				dx.setAt(k, e);
			}
			dx.multiply(1f / pr_sum);

			dist = ArrayMath.cosine(dy.values(), dx.values());

			if (dist < 0.00001) {
				break;
			}

			dy = dx.copy();
		}

		doc.setIndexes(dx.indexes());
		doc.setValues(dx.values());
		doc.setSum(dx.sum());
	}

	public static void parsimoniousLanguageModels(Collection<SparseVector> docs) {
		DenseVector cnts_w_in_c = wordCounts(docs);
		cnts_w_in_c.normalize();

		parsimoniousLanguageModels(docs, cnts_w_in_c, 0.5);
	}

	public static void parsimoniousLanguageModels(Collection<SparseVector> docs, DenseVector cnts_w_in_c, double mixture_jm) {
		for (SparseVector doc : docs) {
			parsimoniousLanguageModel(doc, cnts_w_in_c, mixture_jm);
		}
	}

	public static void print(String log) {
		System.out.println(log);
	}

	public static double tf(double cnt_w) {
		return Math.log(cnt_w) + 1;
	}

	public static void tfidf(Collection<SparseVector> docs) {
		tfidf(docs, documentFreqs(docs));
	}

	public static void tfidf(Collection<SparseVector> docs, DenseVector docFreqs) {

		if (print_log) {
			print("compute tfidfs.");
		}

		double norm = 0;
		int w = 0;
		double cnt = 0;
		double doc_freq = 0;
		double num_docs = docs.size();
		double weight = 0;

		for (SparseVector doc : docs) {
			norm = 0;
			for (int j = 0; j < doc.size(); j++) {
				w = doc.indexAt(j);
				cnt = doc.valueAt(j);
				doc_freq = docFreqs.value(w);
				weight = tfidf(cnt, num_docs, doc_freq);
				norm += (weight * weight);
				doc.setAt(j, weight);
			}
			norm = Math.sqrt(norm);
			doc.multiply(1f / norm);
		}
	}

	public static double tfidf(double cnt_w, double num_docs, double doc_freq) {
		return tf(cnt_w) * idf(num_docs, doc_freq);
	}

	/**
	 * 
	 * Zhai, C., & Lafferty, J. (2002). Two-stage language models for information retrieval. In Proceedings of the 25th annual international
	 * ACM SIGIR conference on Research and development in information retrieval - SIGIR ’02 (p. 49). New York, New York, USA: ACM Press.
	 * http://doi.org/10.1145/564376.564387
	 * 
	 * @param doc
	 * @param cnts_w_in_c
	 * @param prior_dirichlet
	 * @param cnts_w_in_bg
	 * @param mixture_jm
	 */
	public static void twoStageLanguageModel(SparseVector doc, DenseVector cnts_w_in_c, double prior_dir, DenseVector cnts_w_in_bg,
			double mixture_jm) {
		double pr_sum = 0;

		for (int j = 0; j < doc.size(); j++) {
			int w = doc.indexAt(j);
			double pr_w_in_d = twoStageSmoothing(w, doc, cnts_w_in_c, prior_dir, cnts_w_in_bg, mixture_jm);
			pr_sum += pr_w_in_d;
			doc.setAt(j, pr_w_in_d);
		}
		doc.setSum(pr_sum);
	}

	public static void twoStageLanguageModels(List<SparseVector> docs) {
		DenseVector cnts_w_in_c = wordCounts(docs);
		twoStageLanguageModels(docs, cnts_w_in_c, prior_dirichlet, cnts_w_in_c, mixture_jm);
	}

	public static void twoStageLanguageModels(List<SparseVector> docs, DenseVector cnts_w_in_c, double prior_dir, DenseVector cnts_w_in_bg,
			double mixture_jm) {
		for (SparseVector doc : docs) {
			twoStageLanguageModel(doc, cnts_w_in_c, prior_dir, cnts_w_in_bg, mixture_jm);
		}
	}

	public static double twoStageSmoothing(double cnt_w_in_d, double len_d, double pr_w_in_c, double prior_dir, double pr_w_in_bg,
			double mixture_jm) {
		double pr_w_in_d = len_d > 0 ? cnt_w_in_d / len_d : 0;
		double pr_w_in_d_dir = dirichletSmoothing(pr_w_in_d, len_d, pr_w_in_c, prior_dir);
		double pr_w_in_d_jm = jelinekMercerSmoothing(pr_w_in_d_dir, pr_w_in_bg, mixture_jm);
		return pr_w_in_d_jm;
	}

	public static double twoStageSmoothing(int w, SparseVector doc, DenseVector cnts_w_in_c, double prior_dir, DenseVector cnts_w_in_qbg,
			double mixture_jm) {
		double cnt_w_in_d = doc.prob(w);
		double pr_w_in_c = cnts_w_in_c.prob(w);
		double pr_w_in_qbg = cnts_w_in_qbg.prob(w);
		double len_d = doc.sum();
		return twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_qbg, mixture_jm);
	}

	// public static void tf_rf(List<SparseVector> xs, int termSize) {
	// System.out.println("weight by tf-rf");
	// int[][] term_doc = makeInvertedIndex(xs, termSize);
	// ListMap<Integer, Integer> label_doc = groupByLabel(xs);
	//
	// double N = xs.size();
	//
	// for (int labelId : label_doc.keySet()) {
	// Set<Integer> termSet = new HashSet<Integer>();
	// List<Integer> docIds4Label = label_doc.get(labelId);
	//
	// for (int docId : docIds4Label) {
	// SparseVector x = xs.get(docId);
	// for (int i = 0; i < x.size(); i++) {
	// int termId = x.indexAtLoc(i);
	// termSet.add(termId);
	// }
	// }
	//
	// SparseVector term_rf = new SparseVector(termSet.size());
	// int loc = 0;
	//
	// for (int termId : termSet) {
	// int[] docIds4Term = term_doc[termId];
	// double N11 = 0;
	// double N01 = 0;
	// double N10 = 0;
	// double N00 = 0;
	//
	// int loc1 = 0, loc2 = 0;
	//
	// while (loc1 < docIds4Label.size() && loc2 < docIds4Term.length) {
	// int docId1 = docIds4Label.get(loc1);
	// int docId2 = docIds4Term[loc2];
	//
	// if (docId1 == docId2) {
	// N11++;
	// loc1++;
	// loc2++;
	// } else if (docId1 > docId2) {
	// N01++;
	// loc2++;
	// } else if (docId1 < docId2) {
	// N10++;
	// loc1++;
	// }
	// }
	//
	// N00 = N - (N11 + N01 + N10);
	// double chisquare = CommonMath.chisquare(N11, N10, N01, N00);
	// double rf = CommonMath.log2(2 + N11 / (Math.max(1, N01)));
	// term_rf.setAtLoc(loc++, termId, rf);
	// }
	//
	// term_rf.sortByIndex();
	//
	// for (int docId : docIds4Label) {
	// SparseVector x = xs.get(docId);
	// for (int i = 0; i < x.size(); i++) {
	// int termId = x.indexAtLoc(i);
	// double count = x.valueAtLoc(i);
	// double rf = term_rf.valueAlways(termId);
	// x.setAtLoc(i, count * rf);
	// }
	//
	// VectorMath.unitVector(x);
	// }
	// }
	// }

	public static double twoStageSmoothing(int w, SparseVector doc, Vocab vocab, double prior_dir, double mixture_jm) {
		double cnt_w_in_d = doc.value(w);
		double pr_w_in_c = vocab.getProb(w);
		double len_d = doc.sum();
		return twoStageSmoothing(cnt_w_in_d, len_d, pr_w_in_c, prior_dir, pr_w_in_c, mixture_jm);
	}

	public static DenseVector wordCounts(Collection<SparseVector> docs) {
		int max_word_id = maxWordIndex(docs) + 1;
		DenseVector ret = new DenseVector(max_word_id);
		for (SparseVector doc : docs) {
			for (int i = 0; i < doc.size(); i++) {
				ret.add(doc.indexAt(i), doc.valueAt(i));
			}
		}
		return ret;
	}

}
