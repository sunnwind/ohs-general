package ohs.naver.cluster;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import ohs.corpus.dump.TextDumper;
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
			List<String> lines = Generics.newLinkedList();
			int file_loc = 0;
			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				File file = files.get(file_loc);

				TextFileReader reader = new TextFileReader(file.getPath(), "euc-kr");

				int num_docs = 0;
				while (reader.hasNext()) {
					String line = reader.next().trim();

					if (line.length() == 0) {
						continue;
					}

					if (lines.size() == batch_size) {
						DecimalFormat df = new DecimalFormat("000000");
						String outFileName = String.format("%s/%s.txt.gz", outPathName, df.format(batch_cnt.getAndIncrement()));
						FileUtils.writeStringCollectionAsText(outFileName, lines);
						lines.clear();
					} else {
						line = StrUtils.wrap(line);
						lines.add(line);
					}
				}
				reader.close();

				if (lines.size() > 0) {
					DecimalFormat df = new DecimalFormat("000000");
					String outFileName = String.format("%s/%s.txt.gz", outPathName, df.format(batch_cnt.getAndIncrement()));
					FileUtils.writeStringCollectionAsText(outFileName, lines);
					lines.clear();
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
			if (file.getName().contains("2016.01_Title_NNVP_pos.dat")) {
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
