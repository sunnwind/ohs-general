package ohs.ir.medical.query;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.types.generic.CounterMap;
import ohs.utils.Generics;

public class QueryReader {

	public static List<BaseQuery> filter(List<BaseQuery> bqs, CounterMap<String, String> relData) {
		List<BaseQuery> ret = Generics.newArrayList();

		for (BaseQuery bq : bqs) {
			if (relData.sizeOfCounter(bq.getId()) > 0) {
				ret.add(bq);
			}
		}
		System.out.printf("filter out queries which have no relevance judgements [%d -> %d].\n", bqs.size(),
				ret.size());
		return ret;
	}

	public static void main(String[] args) throws Exception {

		// {
		// List<BaseQuery> bqs = readClefEHealthQueries(MIRPath.CLEF_EH_2016_QUERY_FILE,
		// null);
		// System.out.println(bqs.size());
		// }

		// {
		// List<BaseQuery> bqs =
		// readTrecGenomicsQueries(MIRPath.TREC_GENO_2007_QUERY_FILE);
		//
		// for (int i = 0; i < bqs.size(); i++) {
		// System.out.println(bqs.get(i));
		// }
		// }

		{
			List<BaseQuery> bqs = readTrecPmQueries(MIRPath.TREC_PM_2017_QUERY_FILE);
			System.out.println(bqs.size());
		}

	}

	public static List<BaseQuery> readClefEHealthQueries(String queryFileName) throws Exception {
		return readClefEHealthQueries(queryFileName, null);
	}

	public static List<BaseQuery> readClefEHealthQueries(String queryFileName, String dischargeDirName)
			throws Exception {
		List<BaseQuery> ret = Generics.newArrayList();

		Document doc = Jsoup.parse(FileUtils.readFromText(queryFileName));

		int year = 2013;

		if (queryFileName.contains("clef2014")) {
			year = 2014;
		} else if (queryFileName.contains("clef2015")) {
			year = 2015;
		} else if (queryFileName.contains("2016")) {
			year = 2016;
		}

		Elements elems = null;

		if (year == 2016) {
			elems = doc.getElementsByTag("query");

			String[] tags = { "id", "title" };

			for (int i = 0; i < elems.size(); i++) {
				Element elem = elems.get(i);
				String[] vals = new String[tags.length];

				for (int j = 0; j < tags.length; j++) {
					String s = elem.getElementsByTag(tags[j]).get(0).text();
					if (s.length() > 0) {
						vals[j] = s;
					}
				}

				String id = vals[0];
				String dischargeFileName = "";
				String discharge = "";
				String title = "";
				String description = vals[1];
				String profile = "";
				String narrative = "";

				ClefEHealthQuery cq = new ClefEHealthQuery(id, discharge, title, description, profile, narrative);
				ret.add(cq);
			}
		} else if (year == 2015) {
			elems = doc.getElementsByTag("top");

			String[] tags = { "num", "query" };

			for (int i = 0; i < elems.size(); i++) {
				Element elem = elems.get(i);

				String[] vals = new String[tags.length];

				for (int j = 0; j < tags.length; j++) {
					String s = elem.getElementsByTag(tags[j]).get(0).text();
					if (s.length() > 0) {
						vals[j] = s;
					}
				}

				String id = vals[0];
				String dischargeFileName = "";
				String discharge = "";
				String title = "";
				String description = vals[1];
				String profile = "";
				String narrative = "";

				ClefEHealthQuery cq = new ClefEHealthQuery(id, discharge, title, description, profile, narrative);
				ret.add(cq);
			}
		} else {
			if (year == 2013) {
				elems = doc.getElementsByTag("query");
			} else if (year == 2014) {
				elems = doc.getElementsByTag("topic");
			}

			Map<String, File> dischargeFileMap = new TreeMap<String, File>();

			if (dischargeDirName != null) {
				File[] files = new File(dischargeDirName).listFiles();
				for (File file : files) {
					dischargeFileMap.put(file.getName(), file);
				}
			}

			String[] tags = { "id", "discharge_summary", "title", "desc", "profile", "narr" };

			for (int i = 0; i < elems.size(); i++) {
				Element elem = elems.get(i);
				String[] vals = new String[tags.length];
				for (int j = 0; j < tags.length; j++) {
					String s = elem.getElementsByTag(tags[j]).get(0).text();
					if (s.length() > 0) {
						vals[j] = s;
					}
				}

				String id = vals[0];
				String dischargeFileName = vals[1].trim();

				File dischargeFile = dischargeFileMap.get(dischargeFileName);
				String discharge = "";

				if (dischargeFile != null) {
					discharge = FileUtils.readFromText(dischargeFile.getPath());
				} else {
					new FileNotFoundException(dischargeFileName);
				}

				String title = vals[2];
				String description = vals[3];
				String profile = vals[4];
				String narrative = vals[5];

				ClefEHealthQuery cq = new ClefEHealthQuery(id, discharge, title, description, profile, narrative);
				ret.add(cq);
			}
		}

		System.out.printf("read [%d] queries at [%s]\n", ret.size(), queryFileName);

		return ret;
	}

	public static List<BaseQuery> readOhsumedQueries(String fileName) throws Exception {
		/*
		 * .I Sequential identifier
		 * 
		 * .B Patient information
		 * 
		 * .W Information request
		 */

		List<BaseQuery> ret = Generics.newArrayList();
		Map<String, String> map = Generics.newHashMap();
		TextFileReader reader = new TextFileReader(fileName);

		while (reader.hasNext()) {
			String line = reader.next();

			if (line.startsWith(".I")) {
				String[] parts = line.split("[ \t]+");
				String key = parts[0];
				String value = parts[1];

				if (map.size() == 0) {
					map.put(key, value);
				} else {
					String id = map.get(".I");
					String patientInfo = map.get(".B");
					String infoRequest = map.get(".W");

					ret.add(new OhsumedQuery(id, patientInfo, infoRequest));

					map = new HashMap<String, String>();
					map.put(key, value);
				}
			} else {
				reader.hasNext();
				String value = reader.next();
				map.put(line, value);
			}
		}
		reader.close();

		String id = map.get(".I");
		String patientInfo = map.get(".B");
		String infoRequest = map.get(".W");

		ret.add(new OhsumedQuery(id, patientInfo, infoRequest));

		System.out.printf("read [%d] queries at [%s]\n", ret.size(), fileName);

		return ret;
	}

	public static List<BaseQuery> readQueries(String fileName) throws Exception {
		List<BaseQuery> ret = Generics.newArrayList();

		if (fileName.contains("trec_cds")) {
			ret = readTrecCdsQueries(fileName);
		} else if (fileName.contains("clef_ehealth")) {
			ret = readClefEHealthQueries(fileName);
		} else if (fileName.contains("ohsumed")) {
			ret = readOhsumedQueries(fileName);
		} else if (fileName.contains("trec_genomics")) {
			ret = readTrecGenomicsQueries(fileName);
		} else if (fileName.contains("trec_pm")) {
			ret = readTrecPmQueries(fileName);
		}
		return ret;
	}

	public static List<BaseQuery> readTrecCdsQueries(String fileName) throws Exception {
		List<BaseQuery> ret = Generics.newArrayList();

		Document doc = Jsoup.parse(FileUtils.readFromText(fileName));
		Elements elems = doc.getElementsByTag("topic");

		for (int i = 0; i < elems.size(); i++) {
			Element elem2 = elems.get(i);
			String id = elem2.attr("number");
			String type = elem2.attr("type");
			String note = "";
			String desc = "";
			String summary = "";
			String diagnosis = "";

			desc = elem2.getElementsByTag("description").get(0).text();
			summary = elem2.getElementsByTag("summary").get(0).text();

			if (fileName.contains("tred_cds/2015")) {

			} else if (fileName.contains("topic-2015-B.xml")) {
				diagnosis = elem2.getElementsByTag("diagnosis").get(0).text();
			} else if (fileName.contains("topics2016.xml")) {
				note = elem2.getElementsByTag("note").get(0).text();
			}

			ret.add(new TrecCdsQuery(id, type, note, desc, summary, diagnosis));
		}

		System.out.printf("read [%d] queries at [%s]\n", ret.size(), fileName);
		return ret;
	}

	public static List<BaseQuery> readTrecGenomicsQueries(String queryFileName) throws Exception {
		List<BaseQuery> ret = Generics.newArrayList();
		List<String> lines = FileUtils.readLinesFromText(queryFileName);

		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			String id = line.substring(0, 5);
			id = id.substring(1, 4);
			String desc = line.substring(5);
			TrecGenomicsQuery q = new TrecGenomicsQuery(id, desc);
			ret.add(q);
		}
		System.out.printf("read [%d] queries at [%s]\n", ret.size(), queryFileName);
		return ret;
	}

	public static List<BaseQuery> readTrecPmQueries(String fileName) throws Exception {
		List<BaseQuery> ret = Generics.newArrayList();

		Document doc = Jsoup.parse(FileUtils.readFromText(fileName));

		Elements elems = doc.getElementsByTag("topic");

		for (int i = 0; i < elems.size(); i++) {
			Element elem2 = elems.get(i);
			String id = elem2.attr("number");
			String disease = elem2.getElementsByTag("disease").get(0).text();
			String gene = elem2.getElementsByTag("gene").get(0).text();
			String demographic = elem2.getElementsByTag("demographic").get(0).text();
			String other = elem2.getElementsByTag("other").get(0).text();

			if (other.equals("None")) {
				other = "";
			}

			TrecPmQuery q = new TrecPmQuery(id, disease, gene, demographic, other);
			ret.add(q);
		}

		System.out.printf("read [%d] queries at [%s]\n", ret.size(), fileName);
		return ret;
	}
}
