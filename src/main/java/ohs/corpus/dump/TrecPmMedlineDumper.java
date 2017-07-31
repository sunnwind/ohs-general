package ohs.corpus.dump;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPInputStream;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Parser;
import org.jsoup.select.Elements;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class TrecPmMedlineDumper extends TextDumper {

	class Worker implements Callable<Integer> {

		private AtomicInteger file_cnt;

		private List<File> files;

		public Worker(List<File> files, AtomicInteger file_cnt) {
			super();
			this.files = files;
			this.file_cnt = file_cnt;
		}

		@Override
		public Integer call() throws Exception {
			int file_loc = 0;

			while ((file_loc = file_cnt.getAndIncrement()) < files.size()) {
				File inFile = files.get(file_loc);
				File outFile = new File(outPathName, inFile.getName().replace("xml.gz", "txt.gz"));
				List<String> res = Generics.newArrayList();

				String content = "";
				{
					GZIPInputStream gzis = new GZIPInputStream(new FileInputStream(inFile));
					ByteArrayOutputStream baos = new ByteArrayOutputStream();
					byte[] buf = new byte[1024 * 1024];
					int size = 0;
					while ((size = gzis.read(buf)) != -1) {
						baos.write(buf, 0, size);
					}
					gzis.close();

					content = new String(baos.toByteArray());
				}

				Document fileElem = Jsoup.parse(content, "", Parser.xmlParser());
				Elements docElems = fileElem.getElementsByTag("PubmedArticle");

				for (int i = 0; i < docElems.size(); i++) {
					Element docElem = docElems.get(i);

					String pmcid = "";
					String journal = "";
					String title = "";
					String abs = "";
					String meshes = "";
					String chems = "";

					{
						Elements elem = docElem.getElementsByTag("PMID");
						if (elem.size() > 0) {
							pmcid = elem.get(0).text();
						}
					}

					{
						Elements elem = docElem.getElementsByTag("Journal");
						if (elem.size() > 0) {
							Elements elem2 = elem.get(0).getElementsByTag("Title");
							if (elem2.size() > 0) {
								journal = elem2.get(0).text();
							}
						}
					}

					{
						Elements elem = docElem.getElementsByTag("ArticleTitle");
						if (elem.size() > 0) {
							List<String> l = Generics.newArrayList(elem.size());
							for (int j = 0; j < elem.size(); j++) {
								l.add(elem.get(j).text());
							}
							title = StrUtils.join(StrUtils.LINE_REP, l);
						}
					}

					{
						Elements elem = docElem.getElementsByTag("AbstractText");
						if (elem.size() > 0) {
							List<String> l = Generics.newArrayList(elem.size());
							for (int j = 0; j < elem.size(); j++) {
								l.add(elem.get(j).text());
							}

							abs = StrUtils.join(StrUtils.LINE_REP, l);
						}
					}

					{
						List<String> l = Generics.newArrayList();
						Elements elem = docElem.getElementsByTag("MeshHeading");
						if (elem.size() > 0) {
							for (int j = 0; j < elem.size(); j++) {
								l.add(elem.get(j).text());
							}
						}

						meshes = StrUtils.join(StrUtils.LINE_REP, l);
					}

					List<String> vals = Generics.newLinkedList();
					vals.add(pmcid);
					vals.add(journal);
					vals.add(title);
					vals.add(abs);
					vals.add(meshes);
					vals = StrUtils.wrap(vals);
					res.add(StrUtils.join("\t", vals));
				}

				FileUtils.writeStringCollectionAsText(outFile.getPath(), res);
			}

			return (int) 0;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		{
			TrecPmMedlineDumper dh = new TrecPmMedlineDumper(MIRPath.TREC_PM_2017_COL_MEDLINE_RAW_DIR,
					MIRPath.TREC_PM_2017_COL_MEDLINE_LINE_DIR);
			dh.setThreadSize(5);
			dh.dump();

		}

		System.out.println("process ends.");
	}

	private int thread_size = 1;

	public TrecPmMedlineDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		FileUtils.deleteFilesUnder(outPathName);

		List<File> files = Generics.newArrayList();

		for (File file : new File(inPathName).listFiles()) {
			if (file.isDirectory() && file.getName().startsWith("medline")) {
				for (File f : file.listFiles()) {
					if (f.getName().endsWith("xml.gz")) {
						files.add(f);
					}
				}
			}
		}

		Collections.sort(files);

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		AtomicInteger file_cnt = new AtomicInteger(0);

		batch_size = 2000;

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(files, file_cnt)));
		}

		for (int i = 0; i < fs.size(); i++) {
			fs.get(i).get();
		}

		tpe.shutdown();

	}

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}
}
