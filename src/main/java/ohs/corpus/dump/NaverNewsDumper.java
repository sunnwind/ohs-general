package ohs.corpus.dump;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import ohs.fake.FNPath;
import ohs.io.FileUtils;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class NaverNewsDumper extends TextDumper {

	class Worker1 implements Callable<Integer> {

		private AtomicInteger file_cnt;

		private List<File> files;

		public Worker1(List<File> files, AtomicInteger file_cnt) {
			super();
			this.files = files;
			this.file_cnt = file_cnt;
		}

		@Override
		public Integer call() throws Exception {
			int file_loc = 0;
			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				File dir = files.get(file_loc);

				List<File> files = FileUtils.getFilesUnder(dir);
				List<String> res = Generics.newLinkedList();

				int batch_cnt = 0;

				for (int i = 0; i < files.size(); i++) {
					File file = files.get(i);

					Document doc = Jsoup.parse(FileUtils.readFromText(file));
					Elements docElems = doc.getElementsByTag("doc");

					for (int j = 0; j < docElems.size(); j++) {
						Element docElem = docElems.get(j);

						String id = docElem.getElementsByAttributeValue("name", "id").get(0).text();
						String oid = docElem.getElementsByAttributeValue("name", "oid").get(0).text();
						String title = docElem.getElementsByAttributeValue("name", "title").get(0).text();
						String date = docElem.getElementsByAttributeValue("name", "date").get(0).text();
						String cat = docElem.getElementsByAttributeValue("name", "category1").get(0).text();
						String content = docElem.getElementsByAttributeValue("name", "content").get(0).text();
						String url = docElem.getElementsByAttributeValue("name", "url").get(0).text();

						List<String> vals = Generics.newArrayList();
						vals.add(id);
						vals.add(oid);
						vals.add(cat);
						vals.add(date);
						vals.add(title);
						vals.add(content);
						vals.add(url);

						vals = StrUtils.wrap(vals);
						res.add(StrUtils.join("\t", vals));

						if (res.size() % batch_size == 0) {
							write(dir, batch_cnt++, res);
						}
					}
				}

				write(dir, batch_cnt++, res);
			}

			return (int) 0;
		}

		private void write(File dir, int batch_cnt, List<String> res) throws Exception {
			if (res.size() > 0) {
				DecimalFormat df = new DecimalFormat("000000");
				String outFileName = String.format("%s/%s_%s.txt.gz", outPathName, dir.getName().replace(".", ""),
						df.format(batch_cnt));
				FileUtils.writeStringCollectionAsText(outFileName, res);
				res.clear();
			}
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		NaverNewsDumper dh = new NaverNewsDumper(FNPath.NAVER_NEWS_COL_RAW_DIR, FNPath.NAVER_NEWS_COL_LINE_DIR);
		dh.setThreadSize(10);
		dh.setBatchSize(40000);
//		dh.dump();

		System.out.println("process ends.");
	}

	private int thread_size = 3;

	public NaverNewsDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		// FileUtils.deleteFilesUnder(outPathName);

		List<File> files = Generics.newArrayList();

		for (File f1 : new File(inPathName).listFiles()) {
			for (File f2 : f1.listFiles()) {
				if (f2.getName().startsWith("2017.07") || f2.getName().startsWith("2017.08")
						|| f2.getName().startsWith("2017.09"))
					files.add(f2);
			}
		}

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
	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

}
