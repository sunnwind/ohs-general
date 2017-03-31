package ohs.eden.keyphrase.cluster;

import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.io.TextFileReader;
import ohs.io.TextFileWriter;
import ohs.math.ArrayMath;
import ohs.string.search.ppss.Gram;
import ohs.string.search.ppss.GramGenerator;
import ohs.string.sim.EditDistance;
import ohs.string.sim.SequenceFactory;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.types.generic.SetMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class DataAnalyzer {

	class CompareWorker implements Callable<CounterMap<Integer, Integer>> {

		private AtomicInteger cnt;

		private Indexer<String> idx1;

		private Indexer<String> idx2;

		private SetMap<String, Integer> iidx;

		public CompareWorker(Indexer<String> idx1, Indexer<String> idx2, AtomicInteger cnt, SetMap<String, Integer> iindex) {
			this.idx1 = idx1;
			this.idx2 = idx2;
			this.iidx = iindex;
			this.cnt = cnt;
		}

		@Override
		public CounterMap<Integer, Integer> call() throws Exception {

			EditDistance<Character> ed = new EditDistance<Character>();

			CounterMap<Integer, Integer> cm = Generics.newCounterMap();

			Counter<Integer> r = Generics.newCounter();

			int i = 0;

			while ((i = cnt.getAndIncrement()) < idx1.size()) {
				String p1 = idx1.getObject(i);
				String[] ss1 = p1.split("\t");
				ss1 = StrUtils.unwrap(ss1);

				r.clear();

				for (String s : ss1) {
					for (Gram g : gg.generateQGrams(s)) {
						Set<Integer> js = iidx.get(g.getString(), false);

						if (js != null) {
							for (int j : js) {
								r.incrementCount(j, 1);
							}
						}
					}
				}

				List<Integer> js = r.getSortedKeys();
				Counter<Integer> c = Generics.newCounter();

				for (int k = 0; k < js.size() && k < 300; k++) {
					int j = js.get(k);
					String p2 = idx2.getObject(j);
					String[] ss2 = p2.split("\t");
					ss2 = StrUtils.unwrap(ss2);

					double sim1 = ed.getSimilarity(SequenceFactory.newCharSequences(ss1[0], ss2[0]));
					double sim2 = ed.getSimilarity(SequenceFactory.newCharSequences(ss1[1], ss2[1]));
					double sim = ArrayMath.addAfterMultiply(sim1, 0.5, sim2);

					// if (sim >= 0.9) {
					// c.incrementCount(j, sim);
					// }

					if (sim >= 0.9) {
						c.incrementCount(j, sim);
					}
				}

				if (c.size() > 0) {
					cm.setCounter(i, c);
				}
			}

			return cm;
		}
	}

	public static String[] dictFileNames = { "keyword_data.txt.gz", "hanlim_kor-eng.txt", "jst_dict.txt" };

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		DataAnalyzer da = new DataAnalyzer();
		// da.readKeywords();
		// da.compare();

		System.out.println("process ends.");
	}

	private Indexer<String>[] idxs;

	private int thread_size = 50;

	private GramGenerator gg = new GramGenerator(3);

	public void compare() throws Exception {

		int[] locs = new int[] { 2, 1, 0 };

		for (int i = 0; i < locs.length; i++) {
			int loc1 = locs[i];
			for (int j = 0; j < locs.length; j++) {
				int loc2 = locs[j];
				if (loc1 == loc2) {
					continue;
				}

				compare(loc1, loc2);
			}
		}
	}

	public void compare(int qloc, int tloc) throws Exception {
		System.out.printf("map [%s] to [%s].\n", dictFileNames[qloc], dictFileNames[tloc]);

		Indexer<String> qidx = idxs[qloc];
		Indexer<String> tidx = idxs[tloc];

		SetMap<String, Integer> sm = Generics.newSetMap();

		for (int j = 0; j < tidx.size(); j++) {
			String p = tidx.getObject(j);

			for (String s : p.split("\t")) {
				s = StrUtils.unwrap(s);
				for (Gram g : gg.generateQGrams(s)) {
					sm.put(g.getString(), j);
				}
			}
		}

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<CounterMap<Integer, Integer>>> fs = Generics.newArrayList(thread_size);

		AtomicInteger cnt = new AtomicInteger(0);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new CompareWorker(qidx, tidx, cnt, sm)));
		}

		CounterMap<Integer, Integer> cm = Generics.newCounterMap();
		for (int i = 0; i < fs.size(); i++) {
			cm.incrementAll(fs.get(i).get());
		}
		tpe.shutdown();

		// System.out.println(VectorUtils.toCounterMap(cm, idx1, idx2));

		int size1 = qidx.size();
		int size2 = tidx.size();
		double coverage = 1f * cm.size() / size1;

		System.out.printf("size1:\t%d\n", size1);
		System.out.printf("size2:\t%d\n", size2);
		System.out.printf("coverage:\t%f (%d/%d)\n", coverage, cm.size(), size1);

		TextFileWriter writer = new TextFileWriter(KPPath.KYP_DIR + String.format("map_%d-%d.txt", qloc, tloc));

		for (int i = 0; i < qidx.size(); i++) {
			if (!cm.containsKey(i)) {
				continue;
			}

			String[] parts = qidx.get(i).split("\t");
			String q = String.format("(%s)", StrUtils.join(", ", parts));

			StringBuffer sb = new StringBuffer();
			sb.append(q);

			Counter<Integer> c = cm.getCounter(i);

			for (int j = 0; j < tidx.size(); j++) {
				if (!c.containsKey(j)) {
					continue;
				}

				parts = tidx.get(j).split("\t");
				String t = String.format("(%s)", StrUtils.join(", ", parts));

				sb.append(String.format("\t%s:%f", t, c.getCount(j)));
			}

			if (c.size() > 0) {
				writer.write(sb.toString() + "\n");
			}
		}
		writer.close();

	}

	public void readKeywords() throws Exception {
		idxs = new Indexer[dictFileNames.length];

		for (int i = 0; i < dictFileNames.length; i++) {
			Indexer<String> idx = Generics.newIndexer();
			TextFileReader reader = new TextFileReader(KPPath.KYP_DIR + dictFileNames[i], "euc-kr");
			while (reader.hasNext()) {
				String[] parts = reader.next().split("\t");

				if (i < 2 && reader.getLineCnt() == 1) {
					continue;
				}

				if (parts.length < 3) {
					continue;
				}

				String[] ss = null;
				if (i == 0) {
					ss = new String[] { parts[0], parts[1] };
				} else {
					ss = new String[] { parts[1], parts[2] };
					ss = StrUtils.wrap(ss);
				}

				idx.add(StrUtils.join("\t", ss));
			}
			reader.close();
			idxs[i] = idx;
		}
	}

}
