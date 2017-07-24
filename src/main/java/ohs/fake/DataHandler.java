package ohs.fake;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;
import ohs.corpus.type.RawDocumentCollection;
import ohs.io.FileUtils;
import ohs.io.TextFileWriter;
import ohs.math.VectorMath;
import ohs.math.VectorUtils;
import ohs.matrix.SparseVector;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.Indexer;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class DataHandler {

	class PosTaggingWorker implements Callable<Integer> {

		private AtomicInteger file_cnt;

		private List<File> files;

		private Komoran komoran;

		public PosTaggingWorker(Komoran komoran, List<File> files, AtomicInteger file_cnt) {
			super();
			this.komoran = komoran;
			this.files = files;
			this.file_cnt = file_cnt;
		}

		@Override
		public Integer call() throws Exception {
			int file_loc = 0;
			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				File file = files.get(file_loc);

				List<String> lines = FileUtils.readLinesFromText(file);

				for (int i = 0; i < lines.size(); i++) {
					String line = lines.get(i);

					List<String> ps = Generics.newArrayList(line.split("\t"));

					ps = StrUtils.unwrap(ps);

					// vals.add(id);
					// vals.add(oid);
					// vals.add(cat);
					// vals.add(date);
					// vals.add(title);
					// vals.add(content);
					// vals.add(url);

					int j = 0;
					String id = ps.get(j++);
					String oid = ps.get(j++);
					String cat = ps.get(j++);
					String date = ps.get(j++);
					String title = ps.get(j++);
					String content = ps.get(j++);
					String url = ps.get(j++);

					content = content.replace(". ", ".\n\n");

					MDocument taggedTitle = newMDocument(komoran.analyze(title, 1));

					MDocument taggedContent = new MDocument();

					for (String s : content.split("\n")) {
						if (s.length() == 0) {
							continue;
						}

						taggedContent.addAll(newMDocument(komoran.analyze(s, 1)));
					}

					// MDocument taggedContent = getText(komoran.analyze(content, 1));

					ps.clear();
					ps.add(id);
					ps.add(oid);
					ps.add(cat);
					ps.add(date);
					ps.add(taggedTitle.toString().replace("\n", StrUtils.LINE_REP));
					ps.add(taggedContent.toString().replace("\n", StrUtils.LINE_REP));
					ps.add(url);

					ps = StrUtils.wrap(ps);

					lines.set(i, StrUtils.join("\t", ps));
				}

				FileUtils.writeStringCollectionAsText(file.getPath().replace("line", "line_pos"), lines);
			}
			return (int) 0;
		}

	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		DataHandler dh = new DataHandler();
		// dh.tagPOS();
		dh.extractVocab();

		System.out.println("process ends.");
	}

	public void extractVocab() throws Exception {
		RawDocumentCollection rdc = new RawDocumentCollection(FNPath.FN_COL_DC_DIR);

		TextFileWriter writer = new TextFileWriter(FNPath.DATA_DIR + "dissam_docs.txt");
		int cnt = 0;

		for (int i = 0; i < rdc.size(); i++) {
			Map<String, String> m = rdc.getMap(i);

			String id = m.get("id");
			String cat = m.get("cat1");
			String title = m.get("title");
			String content = m.get("body");

			String s = title + "\n" + content.replace(StrUtils.LINE_REP, "\n");
			s = s.trim();

			MDocument md = MDocument.newDocument(s);

			CounterMap<String, Integer> cm = Generics.newCounterMap();
			Indexer<String> wordIdxer = Generics.newIndexer();
			List<String> l = Generics.newArrayList(md.size());

			for (int j = 0; j < md.size(); j++) {
				MSentence ms = md.get(j);

				String type = "T";

				if (j != 0) {
					type = "B";
				}

				Counter<Integer> c = cm.getCounter(type);

				StringBuffer sb = new StringBuffer();

				for (Token t : ms.getTokens()) {
					String word = t.get(0);
					String pos = t.get(1);
					sb.append(word + " ");

					if (pos.startsWith("N")) {
						String ss = String.format("%s / %s", t.get(0), t.get(1));
						int w = wordIdxer.getIndex(ss);
						c.incrementCount(w, 1);
					}
				}
				l.add(sb.toString().trim());
			}

			String text = StrUtils.join("\n", l);
			SparseVector sv1 = new SparseVector(cm.getCounter("T"));
			SparseVector sv2 = new SparseVector(cm.getCounter("B"));

			double cosine = VectorMath.cosine(sv1, sv2);

			if (sv1.size() == 0 || sv2.size() == 0) {
				continue;
			}

			if (cosine < 0.1) {

				// System.out.println("=============");
				// System.out.printf("cosine:\t%f\n", cosine);
				// System.out.println(VectorUtils.toCounter(sv1, wordIdxer));
				// System.out.println(VectorUtils.toCounter(sv2, wordIdxer));
				// System.out.println("------------");
				// System.out.println(text);
				// System.out.println();

				StringBuffer sb = new StringBuffer();
				sb.append(String.format("id:\t%s", id));
				sb.append(String.format("\ndseq:\t%d", i));
				sb.append(String.format("\ncat1:\t%s", cat));
				sb.append("\ntext:");
				sb.append("\n" + text);
				sb.append(String.format("\nTV:\t%s", VectorUtils.toCounter(sv1, wordIdxer)));
				sb.append(String.format("\nBV:\t%s", VectorUtils.toCounter(sv2, wordIdxer)));
				sb.append(String.format("\ncosine:\t%f", cosine));

				writer.write(sb + "\n\n");

				if (++cnt == 50000) {
					break;
				}
			}
		}
		rdc.close();
		writer.close();

	}

	private MDocument newMDocument(List<List<List<Pair<String, String>>>> result) {
		MDocument md = new MDocument();

		for (int i = 0; i < result.size(); i++) {
			List<List<Pair<String, String>>> ll = result.get(i);

			MSentence ms = new MSentence();

			for (int j = 0; j < ll.size(); j++) {
				List<Pair<String, String>> l = ll.get(j);

				MultiToken mt = new MultiToken();

				for (int k = 0; k < l.size(); k++) {
					Pair<String, String> p = l.get(k);
					String f = p.getFirst().replace(" ", "_");
					String s = p.getSecond();

					if (s.length() == 0) {
						continue;
					}

					Token t = new Token();
					t.add(f);
					t.add(s);

					mt.add(t);
				}
				ms.add(mt);
			}
			md.add(ms);
		}

		return md;
	}

	public void tagPOS() throws Exception {
		FileUtils.deleteFilesUnder(FNPath.FN_COL_LINE_POS_DIR);

		AtomicInteger file_cnt = new AtomicInteger(0);

		int thread_size = 3;

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(thread_size);

		List<File> files = FileUtils.getFilesUnder(FNPath.FN_COL_LINE_DIR);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new PosTaggingWorker(new Komoran("lib/models-full/"), files, file_cnt)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}

		tpe.shutdown();
	}

}
