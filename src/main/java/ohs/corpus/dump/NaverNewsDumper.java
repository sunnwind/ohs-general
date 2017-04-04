package ohs.corpus.dump;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class NaverNewsDumper extends TextDumper {

	class Worker1 implements Callable<Integer> {

		private List<File> files;

		private AtomicInteger file_cnt;

		public Worker1(List<File> files, AtomicInteger file_cnt) {
			super();
			this.files = files;
			this.file_cnt = file_cnt;
		}

		@Override
		public Integer call() throws Exception {
			List<String> res = Generics.newLinkedList();
			int file_loc = 0;
			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				File file = files.get(file_loc);

				TextFileReader reader = new TextFileReader(file.getPath(), "euc-kr");

				int num_docs = 0;
				while (reader.hasNext()) {
					List<String> lines = reader.nextLines();

					for (int i = 0; i < lines.size(); i++) {
						lines.set(i, StrUtils.normalizeSpaces(lines.get(i)));
					}

					if (lines.size() < 3) {
						continue;
					}

					if (lines.size() > 3) {
						String title = lines.get(0);
						String section = lines.get(1);
						String body = StrUtils.join(" ", lines, 2, lines.size());

						lines.clear();

						lines.add(title);
						lines.add(section);
						lines.add(body);
					}

					lines = StrUtils.wrap(lines);
					
					res.add(StrUtils.join("\t", lines));

					if (res.size() % batch_size == 0) {
						DecimalFormat df = new DecimalFormat("000000");
						String outFileName = String.format("%s/%s.txt.gz", outPathName, df.format(batch_cnt.getAndIncrement()));
						FileUtils.writeStringCollection(outFileName, res);
						res.clear();
					}
				}
				reader.close();

				if (res.size() > 0) {
					DecimalFormat df = new DecimalFormat("000000");
					String outFileName = String.format("%s/%s.txt.gz", outPathName, df.format(batch_cnt.getAndIncrement()));
					FileUtils.writeStringCollection(outFileName, res);
					res.clear();
				}
			}

			return (int) 0;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		{
			NaverNewsDumper dh = new NaverNewsDumper("../../data/naver_news/col/raw/", "../../data/naver_news/col/line/");
			dh.dump();
		}

		System.out.println("process ends.");
	}

	public NaverNewsDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		FileUtils.deleteFilesUnder(outPathName);

		List<File> files = Generics.newLinkedList();

		for (File file : new File(inPathName).listFiles()) {
			if (file.getName().endsWith(".pos")) {
				files.add(file);
			}
		}

		int thread_size = 3;

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList(thread_size);

		batch_cnt = new AtomicInteger(0);

		AtomicInteger file_cnt = new AtomicInteger(0);

		batch_size = 20000;

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker1(files, file_cnt)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}

		tpe.shutdown();

		// makeSingleFile();
		// extractText();
	}

}
