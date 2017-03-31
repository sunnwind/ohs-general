package ohs.eden.keyphrase.cluster;

import java.util.List;
import java.util.Set;

import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.math.VectorMath;
import ohs.matrix.DenseMatrix;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.common.StrPair;
import ohs.types.generic.Counter;
import ohs.types.number.IntegerArray;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class ClusterValidater {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		KeywordData kwdData = new KeywordData();
		kwdData.readObject(KPPath.KYP_DATA_CLUSTER_SER_FILE);

		ClusterValidater f = new ClusterValidater(kwdData);
		f.validate();

		System.out.println("process ends.");
	}

	private KeywordData kwdData;

	public ClusterValidater(KeywordData kwdData) {
		this.kwdData = kwdData;
	}

	public void validate() throws Exception {
		TextFileWriter writer = new TextFileWriter(KPPath.KYP_DIR + "cluster_filter.txt.gz");
		int filter_cnt = 0;

		IntegerArray toFilter = new IntegerArray();

		for (int cid : kwdData.getClusterToKeywords().keySet()) {
			Set<Integer> kwdids = kwdData.getClusterToKeywords().get(cid);

			List<SparseVector> svs = Generics.newArrayList(kwdids.size());
			List<Integer> idxs = Generics.newArrayList();

			for (int kwdid : kwdids) {
				StrPair kwdp = kwdData.getKeywordIndexer().getObject(kwdid);

				Counter<Integer> c = Generics.newCounter();

				for (String s : kwdp.asArray()) {
					s = StrUtils.normalizePunctuations(s);
					s = StrUtils.normalizeSpaces(s);
					s = s.toLowerCase();

					for (int i = 0; i < s.length(); i++) {
						c.incrementCount((int) s.charAt(i), 1);
					}
				}

				SparseVector sv = new SparseVector(c);
				VectorMath.unitVector(sv);

				idxs.add(kwdid);
				svs.add(sv);
			}

			SparseMatrix sm = new SparseMatrix(idxs, svs);

			DenseMatrix sims = new DenseMatrix(svs.size());

			double avg_sim = 0;
			int cnt = 0;

			for (int i = 0; i < sm.rowSize(); i++) {
				SparseVector sv1 = sm.rowAt(i);
				for (int j = i + 1; j < svs.size(); j++) {
					SparseVector sv2 = sm.rowAt(j);
					double sim = VectorMath.dotProduct(sv1, sv2);
					sims.add(i, j, sim);

					avg_sim += sim;
					cnt++;
				}
			}

			avg_sim /= cnt;

			if (avg_sim >= 0.9 || kwdids.size() == 1) {
				// System.out.println("------------------------");
				// System.out.println(kwdData.getKeywordIndexer().getObject(cid));
				// System.out.println("------------------------");
				// for (int kwdid : kwdids) {
				// StrPair kwdp = kwdData.getKeywordIndexer().getObject(kwdid);
				// System.out.println(kwdp);
				// }
				//
				// System.out.println();

				toFilter.add(cid);

				List<String> l = Generics.newArrayList();
				l.add(kwdData.getKeywordIndexer().getObject(cid).toString());

				for (int kwdid : kwdids) {
					l.add(kwdData.getKeywordIndexer().getObject(kwdid).toString());
				}

				writer.write(++filter_cnt + "\t" + StrUtils.join("\t", l) + "\n\n");
			}
		}
		writer.close();

		toFilter.trimToSize();

		FileUtils.writeIntegers(KPPath.KYP_DIR + "cids_confirmed.ser.gz", toFilter.values());

	}

}
