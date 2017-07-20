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
					byte[] buf = new byte[1024];
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
							title = elem.get(0).text();
						}
					}

					{
						Elements elem = docElem.getElementsByTag("AbstractText");
						if (elem.size() > 0) {
							abs = elem.get(0).text();
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

		// public Integer callOld() throws Exception {
		// String outPath = new File(files.replace(inPathName, outPathName)).getParent().replace("\\", "/");
		// List<String> res = Generics.newArrayList();
		//
		// GZIPInputStream gzis = new GZIPInputStream(new FileInputStream(new File(files)));
		// // GzipCompressorInputStream gzcis = new GzipCompressorInputStream(new FileInputStream(new File(inFileNames)));
		// TarArchiveInputStream tais = new TarArchiveInputStream(gzis);
		// BufferedReader br = new BufferedReader(new InputStreamReader(tais));
		// TarArchiveEntry tae = null;
		//
		// int num_docs = 0;
		// // read every single entry in TAR file
		// while ((tae = tais.getNextTarEntry()) != null) {
		// // the following two lines remove the .tar.gz extension for the folder name
		// // System.out.println(entry.getName());
		//
		// // if (num_docs_in_coll == 40000) {
		// // break;
		// // }
		//
		// if (tae.isDirectory()) {
		// continue;
		// }
		//
		// String fileName = tae.getName();
		//
		// if (!fileName.endsWith("xml.gz")) {
		// continue;
		// }
		//
		// StringBuffer sb = new StringBuffer();
		// ByteArrayOutputStream bout = new ByteArrayOutputStream();
		//
		// // {
		// // //
		// // //
		// // String line = null;
		// // while ((line = br.readLine()) != null) {
		// // sb.append(line + "\n");
		// // }
		// // System.out.println(sb.toString());
		// // }
		//
		// // {
		// // GZIPInputStream gzis2 = new GZIPInputStream(new FileInputStream(tae.getFile()));
		// //
		// // byte[] buf = new byte[1024];
		// // int c;
		// // while ((c = gzis2.read(buf)) != -1) {
		// // bout.write(buf, 0, c);
		// // }
		// // String s = new String(bout.toByteArray());;
		// // System.out.println(s);
		// // }
		//
		// {
		// int size = (int) tae.getSize();
		// byte[] buf = new byte[size];
		// tais.read(buf);
		//
		// }
		//
		// if (sb.length() > 0) {
		// Document doc = Jsoup.parse(sb.toString(), "", Parser.xmlParser());
		//
		// String pmcid = "";
		// String title = "";
		// String abs = "";
		// String body = "";
		// String kwds = "";
		//
		// Elements elem1 = doc.getElementsByAttributeValue("pub-id-type", "pmc");
		//
		// if (elem1.size() > 0) {
		// pmcid = elem1.get(0).text();
		//
		// }
		//
		// String[] tags = { "article-title", "abstract", "body", "kwd" };
		//
		// for (int i = 0; i < tags.length; i++) {
		// Elements elem3 = doc.getElementsByTag(tags[i]);
		//
		// if (elem3.size() > 0) {
		// if (i == 0) {
		// title = elem3.get(0).text();
		// } else if (i == 1) {
		// abs = elem3.get(0).text();
		// } else if (i == 2) {
		// body = elem3.get(0).text();
		// } else if (i == 3) {
		// List<String> l = Generics.newArrayList();
		// for (int j = 0; j < elem3.size(); j++) {
		// l.add(elem3.get(j).text());
		// }
		// if (l.size() > 0) {
		// kwds = StrUtils.join(StrUtils.LINE_REP, l);
		// }
		// }
		// }
		// }
		//
		// List<String> values = Generics.newLinkedList();
		// values.add(pmcid);
		// values.add(title);
		// values.add(abs);
		// values.add(body);
		// values.add(kwds);
		//
		// for (int i = 0; i < values.size(); i++) {
		// values.set(i, StrUtils.normalizeSpaces(values.get(i)));
		// }
		//
		// values = StrUtils.wrap(values);
		//
		// res.add(StrUtils.join("\t", values));
		//
		// if (res.size() % batch_size == 0) {
		// DecimalFormat df = new DecimalFormat("00000");
		// String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(file_cnt.getAndIncrement()));
		// FileUtils.writeStringCollectionAsText(outFileName, res);
		// res.clear();
		// }
		// }
		// }
		// tais.close();
		//
		// if (res.size() > 0) {
		// DecimalFormat df = new DecimalFormat("00000");
		// String outFileName = String.format("%s/%s.txt.gz", outPath, df.format(file_cnt.getAndIncrement()));
		// FileUtils.writeStringCollectionAsText(outFileName, res);
		// res.clear();
		// }
		//
		// return (int) 0;
		// }
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		{
			TrecPmMedlineDumper dh = new TrecPmMedlineDumper(MIRPath.TREC_PM_2017_COL_MEDLINE_RAW_DIR,
					MIRPath.TREC_PM_2017_COL_MEDLINE_LINE_DIR);
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
