package ohs.bioasq;

import java.util.List;

import ohs.corpus.type.EnglishNormalizer;
import ohs.corpus.type.EnglishTokenizer;
import ohs.corpus.type.StringTokenizer;
import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.general.NLPUtils;
import ohs.ir.search.app.DocumentSearcher;
import ohs.ir.search.app.LemmaExpander;
import ohs.ir.weight.TermWeighting;
import ohs.math.VectorUtils;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.types.generic.Counter;
import ohs.types.generic.ListMap;
import ohs.types.generic.Pair;
import ohs.types.generic.Vocab;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.Generics;
import ohs.utils.StrUtils;
import ohs.utils.Timer;

public class MeshSearcher {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		MeshSearcher m = new MeshSearcher();
		m.search();

		System.out.println("process ends.");
	}

	private DocumentSearcher ds;

	private StringTokenizer st;

	private MeshTree mt;

	public MeshSearcher() throws Exception {
		ds = new DocumentSearcher(MIRPath.BIOASQ_COL_INDEX_DIR, MIRPath.STOPWORD_INQUERY_FILE);
		ds.setMaxMatchSize(10000);
		ds.setUseIdfMatch(true);

		mt = new MeshTree(MIRPath.BIOASQ_MESH_TREE_SER_FILE);
		st = new EnglishTokenizer();
	}

	public void search() throws Exception {
		Vocab vocab = ds.getVocab();

		List<SparseVector> qData = Generics.newArrayList();
		List<SparseVector> dData = Generics.newArrayList();
		List<SparseVector> mData = Generics.newArrayList();

		LemmaExpander le = new LemmaExpander(vocab, FileUtils.readStringHashMapFromText(MIRPath.TREC_CDS_2014_DIR + "lemmas.txt"));

		String[] testFileNames = { "Task5aDryRun_1raw.json", "Task5aDryRun_2raw.json" };

		int top_k = 200;

		for (int i = 0; i < testFileNames.length; i++) {
			Timer timer = Timer.newTimer();

			String testFileName = testFileNames[i];

			testFileName = MIRPath.BIOASQ_DIR + "test/task5a/" + testFileName;

			ListMap<String, String> testData = TestArticleReader.read(testFileName);
			List<String> pmids = Generics.newArrayList(testData.keySet());

			for (int j = 0; j < pmids.size(); j++) {
				if ((j + 1) % 10 == 0) {
					System.out.printf("[%d/%d, %s]\n", j + 1, testData.size(), timer.stop());
				}

				String pmid = pmids.get(j);
				List<String> items = testData.get(pmid);
				String journal = items.get(0);
				String title = items.get(1);
				String abs = items.get(2);

				Counter<String> c = Generics.newCounter();

				String[] ss = new String[] { title, abs };

				for (int k = 0; k < ss.length; k++) {
					ss[k] = StrUtils.join(" ", st.tokenize(ss[k]));

					for (String word : ss[k].split(" ")) {
						if (ds.getWordFilter().filter(word)) {
							continue;
						}
						double cnt = 1;
						if (k == 0) {
							cnt = 5;
						}
						c.incrementCount(word, cnt);
					}
				}

				SparseVector Q = VectorUtils.toSparseVector(c, vocab);

				SparseVector Q2 = Q.copy();

				for (int k = 0; k < Q2.size(); k++) {
					int w = Q2.indexAt(k);
					double cnt = Q2.valueAt(k);
					double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(w));
					Q2.setAt(k, tfidf);
				}

				Q2.keepTopN(Math.min(Q2.size(), 20));

				System.out.println(Q2.size() + ": " + VectorUtils.toCounter(Q2, vocab));

				SparseVector scores = ds.match(new IntegerArray(Q2.indexes()));

				scores = ds.search(Q, scores).subVector(top_k);

				Counter<Integer> meshes = Generics.newCounter();

				for (int k = 0; k < scores.size(); k++) {
					int docseq = scores.indexAt(k);
					double score = scores.valueAt(k);

					Pair<String, IntegerArrayMatrix> t = ds.getDocumentCollection().getSents(docseq);

					String[] items2 = t.getThird().split("\t");
					items2 = StrUtils.unwrap(items2);

					for (String mesh : items2[2].split("\\|")) {
						int idx = mt.indexOfName(mesh);
						if (idx != -1) {
							meshes.incrementCount(idx, score);
						}
					}
				}

				qData.add(Q);
				dData.add(scores);
				mData.add(new SparseVector(meshes));
			}

			new SparseMatrix(qData).writeObject(MIRPath.BIOASQ_MESH_RES_SEARCH_DIR + String.format("q_kld-%d.ser.gz", i));
			new SparseMatrix(dData).writeObject(MIRPath.BIOASQ_MESH_RES_SEARCH_DIR + String.format("d_kld-%d.ser.gz", i));
			new SparseMatrix(mData).writeObject(MIRPath.BIOASQ_MESH_RES_SEARCH_DIR + String.format("m_kld-%d.ser.gz", i));
		}

		// for (int j = 0; j < testData.size(); j++) {
		// if ((j + 1) % 10 == 0) {
		// System.out.printf("[%d/%d, %s]\n", j + 1, testData.size(), timer.stop());
		// }
		// String[] parts = testData.get(j).split("\t");
		// String pmid = parts[0];
		// String journal = parts[1];
		// String title = parts[2];
		// String abs = parts[3];
		// String meshStr = parts[4];
		//
		// Counter<String> ansMeshes = Generics.newCounter();
		//
		// for (String mesh : meshStr.split("\\|")) {
		// ansMeshes.incrementCount(mesh, 1);
		// }
		//
		// String st = StrUtils.join(" ", NLPUtils.tokenize(title + "\n" + abs));
		// st = sn.normalize(st);
		//
		// SparseVector Q = ds.index(st, false);
		//
		// Counter<Integer> toKeep = Generics.newCounter();
		//
		// for (int k = 0; k < Q.size(); k++) {
		// int w = Q.indexAt(k);
		// double cnt = Q.valueAt(k);
		// double tfidf = TermWeighting.tfidf(cnt, vocab.getDocCnt(), vocab.getDocFreq(w));
		// toKeep.incrementCount(w, tfidf);
		// }
		//
		// toKeep.keepTopNKeys(40);
		//
		// SparseVector Q2 = Q.copy();
		// Q2.pruneExcept(toKeep.keySet());
		//
		// System.out.println(toKeep.size() + ": " + VectorUtils.toCounter(toKeep, vocab));
		//
		// SparseVector scores = ds.match(Q2);
		//
		// SparseVector lm_q = Q.copy();
		// lm_q.normalize();
		//
		// scores = ds.search(lm_q, scores).subVector(top_k);
		//
		// Counter<Integer> meshIdxs = Generics.newCounter();
		//
		// for (int k = 0; k < scores.size(); k++) {
		// int docseq = scores.indexAt(k);
		// double score = scores.valueAt(k);
		//
		// Triple<String, IntegerArrayMatrix, String> t = ds.getDocumentStore().get(docseq);
		//
		// String[] items = t.getThird().split("\t");
		// items = StrUtils.unwrap(items);
		//
		// for (String mesh : items[2].split("\\|")) {
		// int idx = mt.indexOfName(mesh);
		// if (idx != -1) {
		// meshIdxs.incrementCount(idx, score);
		// }
		// }
		// }
		//
		// // System.out.println(journals.toString());
		// // System.out.println(predMeshes.toString());
		// // System.out.println(ansMeshes.toString());
		//
		// qData.add(lm_q);
		// dData.add(scores);
		// mData.add(new SparseVector(meshIdxs));
		// }

		// new SparseMatrix(qData).writeObject(MIRPath.BIOASQ_MESH_RES_SEARCH_DIR + "q_kld.ser.gz");
		// new SparseMatrix(dData).writeObject(MIRPath.BIOASQ_MESH_RES_SEARCH_DIR + "d_kld.ser.gz");
		// new SparseMatrix(mData).writeObject(MIRPath.BIOASQ_MESH_RES_SEARCH_DIR + "m_kld.ser.gz");
	}

}
