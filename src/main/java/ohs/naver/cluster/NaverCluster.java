package ohs.naver.cluster;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.DocumentCollectionCreator;
import ohs.corpus.type.RawDocumentCollection;
import ohs.io.TextFileWriter;
import ohs.ir.weight.TermWeighting;
import ohs.matrix.SparseMatrix;
import ohs.matrix.SparseVector;
import ohs.ml.cluster.KMeansClustering;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.Pair;
import ohs.types.number.IntegerArray;
import ohs.types.number.IntegerArrayMatrix;
import ohs.utils.DataSplitter;
import ohs.utils.Generics;

public class NaverCluster {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		{
			NaverNewsDumper dh = new NaverNewsDumper("../../data/naver_news/col/raw/", "../../data/naver_news/col/line/");
			dh.dump();
		}

		{
			String[] attrs = { "title" };
			String inDir = "../../data/naver_news/col/line/";
			String outDir = "../../data/naver_news/col/dc/";
			boolean append = false;
			NaverRawDocumentCollectionCreator dcc = new NaverRawDocumentCollectionCreator(outDir, append);
			dcc.addAttrs(Generics.newArrayList(attrs));
			NaverRawDocumentCollectionCreator.createFromTokenizedData(dcc, inDir);
			dcc.close();
		}

		{
			DocumentCollectionCreator dcc = new DocumentCollectionCreator();
			dcc.setBatchSize(200);
			dcc.setCountingThreadSize(10);
			dcc.setIndexingThreadSize(5);

			dcc.setMinWordCnt(0);
			dcc.create("../../data/naver_news/col/dc/", -1, new int[] { 0 });
		}

		{
			NaverCluster nc = new NaverCluster();
			nc.cluster();
		}

		System.out.println("process ends.");
	}

	public void cluster() throws Exception {
		RawDocumentCollection rdc = new RawDocumentCollection("../../data/naver_news/col/dc");
		DocumentCollection dc = new DocumentCollection("../../data/naver_news/col/dc");
		List<SparseVector> dvs = Generics.newArrayList(500000);

		int[][] ranges = BatchUtils.getBatchRanges(dc.size(), 20000);

		for (int i = 0; i < ranges.length; i++) {
			int[] range = ranges[i];
			for (Pair<String, IntegerArray> p : dc.getRange(range)) {
				Counter<Integer> c = Generics.newCounter();
				for (int w : p.getSecond()) {
					if (w == DocumentCollection.SENT_END) {
						continue;
					}
					c.incrementCount(w, 1);
				}

				SparseVector dv = new SparseVector(c);
				dvs.add(dv);
			}
		}

		for (SparseVector dv : dvs) {
			double norm = 0;
			double doc_cnt = dc.getVocab().getDocCnt();
			for (int j = 0; j < dv.size(); j++) {
				int w = dv.indexAt(j);
				double cnt = dv.valueAt(j);
				double doc_freq = dc.getVocab().getDocFreq(w);
				double weight = TermWeighting.tfidf(cnt, doc_cnt, doc_freq);
				norm += (weight * weight);
				dv.setAt(j, weight);
			}
			norm = Math.sqrt(norm);
			dv.multiply(1f / norm);
		}

		SparseMatrix X = new SparseMatrix(dvs);

		int thread_size = 10;
		int max_iter = 50;
		int cluster_size = 20;

		KMeansClustering kc = new KMeansClustering(X, cluster_size, dc.getVocab().size());
		kc.setThreadSize(thread_size);
		IntegerArray clusters = kc.cluster(max_iter);

		TextFileWriter writer = new TextFileWriter("../../data/naver_news/cluster.txt");

		IntegerArrayMatrix G = DataSplitter.group(clusters);

		Counter<Integer> c = Generics.newCounter();

		for (int i = 0; i < G.size(); i++) {
			c.setCount(i, G.get(i).size());
		}

		List<Integer> cids = c.getSortedKeys();

		writer.write("[Cluster Info]");
		writer.write("\nCID\tCnt\tRatio");

		for (int i = 0; i < cids.size(); i++) {
			int cid = cids.get(i);
			int cnt = (int) c.getCount(cid);
			double prob = c.getProbability(cid);
			writer.write(String.format("\n%d\t%d\t%f", cid, cnt, prob));
		}

		writer.write("\n\n[Weighting: TFIDF]");
		writer.write("\n[Similarity: Cosine]");
		writer.write(String.format("\n[Iterations: %d]", max_iter));
		writer.write(String.format("\n[Threads: %d]", thread_size));
		writer.write("\n\n");

		for (int i : cids) {
			IntegerArray locs = G.get(i);
			Map<Integer, String> m = Generics.newHashMap(locs.size());

			for (int loc : locs) {
				m.put(loc, rdc.get(loc).get(0));
			}

			List<Integer> dids = Generics.newArrayList(m.keySet());
			Collections.sort(dids);

			writer.write(String.format("\n[ClusterID:\t%d]", i));

			for (int j = 0; j < dids.size(); j++) {
				int loc = dids.get(j);
				String title = m.get(loc);
				writer.write(String.format("\n%d\t%s", loc, title));
			}
			writer.write("\n\n");
		}
		writer.close();

	}

}
