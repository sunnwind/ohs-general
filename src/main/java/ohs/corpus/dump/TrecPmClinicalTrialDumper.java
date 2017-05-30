package ohs.corpus.dump;

import java.io.File;
import java.io.FileInputStream;
import java.text.DecimalFormat;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.jsoup.Jsoup;
import org.jsoup.helper.StringUtil;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.parser.Parser;
import org.jsoup.select.Elements;
import org.jsoup.select.NodeTraversor;
import org.jsoup.select.NodeVisitor;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class TrecPmClinicalTrialDumper extends TextDumper {

	class Worker implements Callable<Integer> {

		private List<File> files;

		private AtomicInteger batch_cnt;

		private int[][] ranges;

		public Worker(List<File> files, int[][] ranges, AtomicInteger batch_cnt) {
			super();
			this.files = files;
			this.ranges = ranges;
			this.batch_cnt = batch_cnt;
		}

		@Override
		public Integer call() throws Exception {
			int batch_loc = 0;
			while ((batch_loc = batch_cnt.getAndIncrement()) < ranges.length) {
				int[] range = ranges[batch_loc];
				List<String> res = Generics.newArrayList(range[1] - range[0]);

				for (int i = range[0]; i < range[1]; i++) {
					File file = files.get(i);
					String xmlStr = FileUtils.readFromText(file);
					Document doc = Jsoup.parse(xmlStr, "", Parser.xmlParser());

					String nctid = "";
					String briefTitle = "";
					String officialTitle = "";
					String content = "";
					String kwds = "";
					String meshes = "";

					nctid = doc.getElementsByTag("nct_id").get(0).text();

					{
						Elements elem = doc.getElementsByTag("brief_title");

						if (elem.size() > 0) {
							briefTitle = elem.get(0).text();
						}
					}

					{
						Elements elem = doc.getElementsByTag("official_title");

						if (elem.size() > 0) {
							officialTitle = elem.get(0).text();
						}
					}

					{
						Elements elem = doc.getElementsByTag("textblock");
						List<String> l = Generics.newArrayList();
						if (elem.size() > 0) {

							for (int j = 0; j < elem.size(); j++) {
								l.add(elem.get(j).text());
							}
						}

						content = StrUtils.join(StrUtils.LINE_REP + StrUtils.LINE_REP, l);
					}

					{
						Elements elem = doc.getElementsByTag("keyword");
						if (elem.size() > 0) {
							List<String> l = Generics.newArrayList();
							for (int j = 0; j < elem.size(); j++) {
								l.add(elem.get(j).text());
							}
							kwds = StrUtils.join(StrUtils.LINE_REP, l);
						}
					}

					{
						Elements elem = doc.getElementsByTag("mesh_term");
						if (elem.size() > 0) {
							List<String> l = Generics.newArrayList();
							for (int j = 0; j < elem.size(); j++) {
								l.add(elem.get(j).text());
							}
							meshes = StrUtils.join(StrUtils.LINE_REP, l);
						}
					}

					List<String> vals = Generics.newLinkedList();
					vals.add(nctid);
					vals.add(briefTitle);
					vals.add(officialTitle);
					vals.add(content);
					vals.add(kwds);
					vals.add(meshes);
					vals = StrUtils.wrap(vals);
					res.add(StrUtils.join("\t", vals));
				}

				DecimalFormat df = new DecimalFormat("00000");
				String outFileName = String.format("%s/%s.txt.gz", outPathName, df.format(batch_loc));
				FileUtils.writeStringCollectionAsText(outFileName, res);
			}

			return (int) 0;
		}
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		TrecPmClinicalTrialDumper dh = new TrecPmClinicalTrialDumper(MIRPath.TREC_PM_2017_COL_CLINICAL_RAW_DIR,
				MIRPath.TREC_PM_2017_COL_CLINICAL_LINE_DIR);
		dh.setThreadSize(3);
		dh.dump();

		System.out.println("process ends.");
	}

	public TrecPmClinicalTrialDumper(String inDirName, String outDirName) {
		super(inDirName, outDirName);
	}

	private int thread_size = 1;

	public void setThreadSize(int thread_size) {
		this.thread_size = thread_size;
	}

	@Override
	public void dump() throws Exception {
		System.out.printf("dump [%s] to [%s].\n", inPathName, outPathName);

		FileUtils.deleteFilesUnder(outPathName);

		List<File> files = Generics.newArrayList();

		for (File file : FileUtils.getFilesUnder(inPathName)) {
			String fileName = file.getName();
			if (fileName.startsWith("NCT") && fileName.endsWith(".xml")) {
				files.add(file);
			}
		}

		Collections.sort(files);

		ThreadPoolExecutor tpe = (ThreadPoolExecutor) Executors.newFixedThreadPool(thread_size);

		List<Future<Integer>> fs = Generics.newArrayList();

		int[][] ranges = BatchUtils.getBatchRanges(files.size(), batch_size);

		for (int i = 0; i < thread_size; i++) {
			fs.add(tpe.submit(new Worker(files, ranges, batch_cnt)));
		}

		for (int i = 0; i < thread_size; i++) {
			fs.get(i).get();
		}

		tpe.shutdown();
	}
}
