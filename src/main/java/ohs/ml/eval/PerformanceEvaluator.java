package ohs.ml.eval;

import java.util.List;
import java.util.Set;

import ohs.matrix.DenseMatrix;
import ohs.matrix.DenseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.Indexer;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerMatrix;
import ohs.utils.Generics;

public class PerformanceEvaluator {

	private Counter<Integer> countCommonLabelSeqs(List<String> anss, List<String> preds, Indexer<String> labelIdxer) {
		Counter<Integer> c = Generics.newCounter(labelIdxer.size());

		int size = preds.size();

		for (int i = 0; i < size;) {
			String pred = preds.get(i);
			String ans = anss.get(i);

			String[] p1 = splitBioLabel(pred);
			String[] a1 = splitBioLabel(ans);

			if (p1[0].equals(a1[0]) && p1[1].equals("B") && a1[1].equals("B")) {
				int e1 = i + 1;
				int e2 = i + 1;

				while (e1 < size) {
					String[] p2 = splitBioLabel(preds.get(e1));
					if (!p2[1].equals("I")) {
						break;
					}
					e1++;
				}

				while (e2 < size) {
					String[] a2 = splitBioLabel(anss.get(e2));
					if (!a2[1].equals("I")) {
						break;
					}
					e2++;
				}

				if (e1 == e2) {
					String label = p1[0];
					c.incrementCount(labelIdxer.indexOf(label), 1);
				}
				i = e1;
			} else {
				i++;
			}
		}
		return c;
	}

	private Set<String> getLabelSeqs(List<String> tags, Indexer<String> labelIdxer) {
		Set<String> ret = Generics.newHashSet();

		for (int i = 0; i < tags.size();) {
			String tag1 = tags.get(i);

			if (tag1.startsWith("B-") || tag1.startsWith("S-")) {
				String prefix1 = tag1.split("-")[0];
				String cat1 = tag1.split("-")[1];

				int match_len = 1;

				if (prefix1.equals("S")) {

				} else if (prefix1.equals("B")) {
					for (int j = i + 1; j < tags.size(); j++) {
						String tag2 = tags.get(j);
						if (tag2.equals("O")) {
							break;
						} else if (tag2.startsWith("B-") || tag2.startsWith("S-")) {
							break;
						}
						match_len++;
					}
				} else {
					System.err.println("wrong tag sequence");
					System.exit(0);
				}
				ret.add(String.format("%d_%d_%s", i, match_len, cat1));
				i += match_len;
			} else {
				i++;
			}
		}
		return ret;
	}

	public Performance evalute(DenseMatrix Y, DenseMatrix Yh, Indexer<String> labelIdxer) {
		return evalute(Y.toDenseVector(), Yh.toDenseVector(), labelIdxer);
	}

	public Performance evalute(DenseVector Y, DenseVector Yh, Indexer<String> labelIdxer) {
		Set<Integer> labels = Generics.newTreeSet();

		if (labelIdxer == null) {
			for (double y : Y.values()) {
				labels.add((int) y);
			}
		} else {
			for (int i = 0; i < labelIdxer.size(); i++) {
				labels.add(i);
			}
		}

		DenseVector corCnts = new DenseVector(labels.size());
		DenseVector predCnts = new DenseVector(labels.size());
		DenseVector ansCnts = new DenseVector(labels.size());

		for (int i = 0; i < Y.size(); i++) {
			int ans = (int) Y.value(i);
			int pred = (int) Yh.value(i);

			ansCnts.add(ans, 1);
			predCnts.add(pred, 1);

			if (ans == pred) {
				corCnts.add(ans, 1);
			}
		}

		if (labelIdxer != null) {
			int idx = labelIdxer.indexOf("O");
			if (idx >= 0) {
				corCnts.set(idx, 0);
				predCnts.set(idx, 0);
				ansCnts.set(idx, 0);

				corCnts.summation();
				predCnts.summation();
				ansCnts.summation();
			}
		}

		Performance p = new Performance(ansCnts, predCnts, corCnts, labelIdxer);
		return p;
	}

	public Performance evaluteSequences(DenseMatrix Y, DenseMatrix Yh, Indexer<String> tagIdxer) {

		int size = Y.sizeOfEntries();

		Indexer<String> catIdxer = Generics.newIndexer();

		{
			Counter<String> c = Generics.newCounter();

			for (String tag : tagIdxer) {
				if (!tag.equals("O")) {
					String[] ps = tag.split("-");
					String cat = ps[1];
					c.incrementCount(cat, 1);
				}
			}

			if (c.size() > 0) {
				catIdxer.addAll(Generics.newTreeSet(c.keySet()));
				catIdxer.add("O");
			} else {
				catIdxer = Generics.newIndexer(tagIdxer);
			}
		}

		DenseVector ansCnts = new DenseVector(catIdxer.size());
		DenseVector predCnts = new DenseVector(catIdxer.size());
		DenseVector corCnts = new DenseVector(catIdxer.size());

		for (int i = 0; i < Y.size(); i++) {
			DenseVector y = Y.get(i);
			DenseVector yh = Yh.get(i);

			List<String> anss = Generics.newArrayList(y.size());
			List<String> preds = Generics.newArrayList(y.size());

			for (int j = 0; j < y.size(); j++) {
				anss.add(tagIdxer.getObject((int) y.value(j)));
				preds.add(tagIdxer.getObject((int) yh.value(j)));
			}

			Set<String> s1 = getLabelSeqs(anss, catIdxer);
			Set<String> s2 = getLabelSeqs(preds, catIdxer);
			Set<String> s3 = Generics.newHashSet(s1.size());

			for (String ans : s2) {
				if (s1.contains(ans)) {
					s3.add(ans);
				}
			}

			for (String tag : s1) {
				String[] ps = tag.split("_");
				ansCnts.add(catIdxer.indexOf(ps[2]), 1);
			}

			for (String tag : s2) {
				String[] ps = tag.split("_");
				predCnts.add(catIdxer.indexOf(ps[2]), 1);
			}

			for (String tag : s3) {
				String[] ps = tag.split("_");
				corCnts.add(catIdxer.indexOf(ps[2]), 1);
			}
		}

		return new Performance(ansCnts, predCnts, corCnts, catIdxer);
	}

	private String[] splitBioLabel(String bioLabel) {
		String[] ps = bioLabel.split("-");
		String label = "";
		String bio = "O";
		if (ps.length == 2) {
			label = ps[0];
			bio = ps[1];
			if (label.equals("B") || label.equals("I")) {
				label = ps[1];
				bio = ps[0];
			}
		}
		return new String[] { label, bio };
	}

}