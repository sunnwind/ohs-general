package ohs.fake;

import java.io.File;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import kr.co.shineware.nlp.komoran.core.analyzer.Komoran;
import kr.co.shineware.util.common.model.Pair;
import ohs.io.FileUtils;
import ohs.nlp.ling.types.MDocument;
import ohs.nlp.ling.types.MSentence;
import ohs.nlp.ling.types.MultiToken;
import ohs.nlp.ling.types.Token;
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
		dh.tagPOS();

		System.out.println("process ends.");
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
					String f = p.getFirst();
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

		int thread_size = 5;

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
