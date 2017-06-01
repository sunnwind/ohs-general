package ohs.eden.keyphrase.mine;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ohs.corpus.type.RawDocumentCollection;
import ohs.corpus.type.EnglishNormalizer;
import ohs.corpus.type.StringNormalizer;
import ohs.eden.keyphrase.cluster.KPPath;
import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.io.TextFileWriter;
import ohs.ir.medical.general.MIRPath;
import ohs.ml.neuralnet.com.BatchUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.types.generic.ListMapMap;
import ohs.utils.Generics;
import ohs.utils.Generics.ListType;
import ohs.utils.StrUtils;

public class UserKeywordCollector {

	public static List<String> getKeywords(String keywordStr) {
		List<String> ret = Generics.newArrayList();
		for (String kw : keywordStr.split(";")) {
			kw = kw.trim();
			if (kw.length() > 0) {
				ret.add(kw);
			}
		}
		return ret;
	}

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");
		UserKeywordCollector dh = new UserKeywordCollector();
		dh.get3PKeywords();
		// dh.getSnomedConcepts();
		// dh.getSnomedPhrases();
		// dh.getMeSHes();
		// dh.getPaperKeywords();

		// dh.getBioASQMeshes();
		// dh.getTrecPmClinicalKeywords();
		// dh.getScopusKeywords();
		// dh.getTrecCdsKeywords();
		// dh.getWikiPhrases();
		// dh.mergeKeywords();
		// dh.getMedicalPhrases();

		System.out.println("process ends.");
	}

	public void get3PKeywords() throws Exception {
		RawDocumentCollection rdc = new RawDocumentCollection(KPPath.COL_DC_DIR);

		Counter<String> c = Generics.newCounter();

		for (int i = 0; i < rdc.size(); i++) {
			int progress = BatchUtils.progress(i + 1, rdc.size());
			if (progress > 0) {
				System.out.printf("[%d percent]\n", progress);
			}

			HashMap<String, String> m = rdc.getMap(i);
			List<String> vals = rdc.get(i);
			int j = 0;
			String type = vals.get(j++);
			String cn = vals.get(j++);
			String korKwsStr = vals.get(j++);
			String engKwsStr = vals.get(j++);

			if (type.equals("patent")) {
				break;
			}

			String[] korKwds = korKwsStr.split("\n");
			String[] engKwds = engKwsStr.split("\n");

			if (korKwds.length == engKwds.length) {
				for (int k = 0; k < korKwds.length; k++) {
					String korKwd = korKwds[k];
					String engKwd = engKwds[k];

					if (korKwd.length() > 0 && engKwd.length() > 0) {
						String[] p = new String[] { korKwd, engKwd };
						p = StrUtils.wrap(p);
						c.incrementCount(StrUtils.join("\t", p), 1);
					}
				}
			}
		}
		FileUtils.writeStringCounterAsText(MIRPath.DATA_DIR + "phrs/3p_kwd.txt", c);
	}

	public void getBioASQMeshes() throws Exception {
		List<File> files = FileUtils.getFilesUnder(MIRPath.BIOASQ_COL_LINE_DIR);
		Counter<String> c = Generics.newCounter();
		for (File file : files) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] ps = line.split("\t");
				ps = StrUtils.unwrap(ps);
				for (String mesh : ps[3].split("\\|")) {
					mesh = StrUtils.normalizeSpaces(mesh);
					if (mesh.length() > 0) {
						c.incrementCount(mesh, 1);
					}
				}
			}
		}
		FileUtils.writeStringCounterAsText(MIRPath.DATA_DIR + "phrs/bioasq_mesh.txt", c);
	}

	public void getMeSHes() throws Exception {

		/***
		 * 
		 * 
		 * AN ANNOTATION
		 * 
		 * AQ ALLOWABLE TOPICAL QUALIFIERS
		 * 
		 * CATSH CATALOGING SUBHEADINGS LIST NAME
		 * 
		 * CX CONSIDER ALSO XREF
		 * 
		 * DA DATE OF ENTRY
		 * 
		 * DC DESCRIPTOR CLASS
		 * 
		 * DE DESCRIPTOR ENTRY VERSION
		 * 
		 * DS DESCRIPTOR SORT VERSION
		 * 
		 * DX DATE MAJOR DESCRIPTOR ESTABLISHED
		 * 
		 * EC ENTRY COMBINATION
		 * 
		 * PRINT ENTRY ENTRY TERM, PRINT **
		 * 
		 * ENTRY ENTRY TERM, NON-PRINT **
		 * 
		 * FX FORWARD CROSS REFERENCE (SEE ALSO REFERENCE)
		 * 
		 * HN HISTORY NOTE
		 * 
		 * MH MESH HEADING
		 * 
		 * MH_TH MESH HEADING THESAURUS ID [= MHTH in ELHILL MESH]
		 * 
		 * MN MESH TREE NUMBER
		 * 
		 * MR MAJOR REVISION DATE
		 * 
		 * MS MESH SCOPE NOTE
		 * 
		 * N1 CAS TYPE 1 NAME
		 * 
		 * OL ONLINE NOTE
		 * 
		 * PA PHARMACOLOGICAL ACTION
		 * 
		 * PI PREVIOUS INDEXING
		 * 
		 * PM PUBLIC MESH NOTE
		 * 
		 * RECTYPE RECORD TYPE [= RY in ELHILL MESH ]
		 * 
		 * RH RUNNING HEAD, MESH TREE STRUCTURES
		 * 
		 * RN CAS REGISTRY/EC NUMBER/UNII CODE
		 * 
		 * RR RELATED REGISTRY NUMBER
		 * 
		 * ST SEMANTIC TYPE *
		 * 
		 * UI UNIQUE IDENTIFIER
		 */

		/**
		 * 
		 * ENTRY and PRINT ENTRY terms may have several subfields, with each subfield entry separated from another by a bar. A final string
		 * "map" indicates which value goes with which subfield. For example,
		 * 
		 * PRINT ENTRY = Avian Sarcoma|T050|T191|NON|EQV|NLM (1994)|930624|abbcdef
		 * 
		 * In the final string of characters - 'abcdeef' - each letter corresponds to a specific subfield and indicates the position of the
		 * subfield and any position repeats:
		 * 
		 * a the term itself
		 * 
		 * b SEMANTIC TYPE*
		 * 
		 * c LEXICAL TYPE*
		 * 
		 * d SEMANTIC RELATION*
		 * 
		 * e THESAURUS ID
		 * 
		 * f DATE
		 * 
		 * s SORT VERSION
		 * 
		 * v ENTRY VERSION
		 * 
		 */

		String sep = " = ";
		Counter<String> c = Generics.newCounter();

		Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);

		{
			TextFileReader reader = new TextFileReader(MIRPath.MESH_COL_RAW_DESCRIPTOR_FILE);
			TextFileWriter writer = new TextFileWriter(MIRPath.MESH_DIR + "word_desc.txt");

			while (reader.hasNext()) {
				List<String> lines = reader.nextLines();
				ListMap<String, String> lm = Generics.newListMap();

				for (int i = 1; i < lines.size();) {
					String l1 = lines.get(i);
					int end = i + 1;

					for (int j = i + 1; j < lines.size(); j++) {
						String l2 = lines.get(j);
						if (l2.contains(sep)) {
							end = j;
							break;
						}
					}

					String s = StrUtils.join(" ", lines, i, end);
					String[] two = s.split(sep);
					String attr = two[0];
					String value = two[1];
					lm.put(attr, value);

					i = end;
				}

				String name = lm.get("MH", true).get(0);

				List<String> items = Generics.newArrayList();

				items.add(name);

				for (String l : lm.get("ENTRY", true)) {
					String[] parts = l.split("\\|");
					items.add(parts[0]);
				}

				for (String l : lm.get("PRINT ENTRY", true)) {
					String[] parts = l.split("\\|");
					items.add(parts[0]);
				}

				Counter<String> c1 = Generics.newCounter();
				for (String item : items) {
					c1.incrementCount(item.toLowerCase(), 1);
				}
				c.incrementAll(c1);

			}
			reader.close();
			writer.close();
		}

		{
			/**
			 * 
			 * Data Element Name
			 * 
			 * DA DATE OF ENTRY
			 * 
			 * FR FREQUENCY
			 * 
			 * HM HEADING MAPPED-TO
			 * 
			 * II INDEXING INFORMATION
			 * 
			 * MR MAJOR REVISION DATE
			 * 
			 * N1 CAS TYPE 1 NAME
			 * 
			 * NM NAME OF SUBSTANCE
			 * 
			 * NM_TH NM TERM THESAURUS ID
			 * 
			 * NO NOTE
			 * 
			 * PA PHARMACOLOGICAL ACTION
			 * 
			 * PI PREVIOUS INDEXING
			 * 
			 * RECTYPE RECORD TYPE [= RY in ELHILL MESH]
			 * 
			 * RN CAS REGISTRY/EC NUMBER/UNII CODE
			 * 
			 * RR RELATED REGISTRY NUMBER
			 * 
			 * SO SOURCE
			 * 
			 * ST SEMANTIC TYPE
			 * 
			 * SY SYNONYM **
			 * 
			 * TH THESAURUS ID
			 * 
			 * UI UNIQUE IDENTIFIER
			 * 
			 * 
			 * 
			 */
			TextFileReader reader = new TextFileReader(MIRPath.MESH_COL_RAW_SUPPLEMENTARY_FILE);

			while (reader.hasNext()) {
				List<String> lines = reader.nextLines();
				ListMap<String, String> lm = Generics.newListMap();

				for (int i = 1; i < lines.size();) {
					String l1 = lines.get(i);
					int end = i + 1;

					for (int j = i + 1; j < lines.size(); j++) {
						String l2 = lines.get(j);
						if (l2.contains(sep)) {
							end = j;
							break;
						}
					}

					String s = StrUtils.join(" ", lines, i, end);
					String[] two = s.split(sep);
					String attr = two[0];
					String value = two[1];
					lm.put(attr, value);

					i = end;
				}

				if (!lm.containsKey("NM")) {
					continue;
				}

				String name = lm.get("NM", true).get(0);

				List<String> items = Generics.newArrayList();

				items.add(name);

				for (String l : lm.get("SY", true)) {
					String[] parts = l.split("\\|");
					items.add(parts[0]);
				}

				for (String l : lm.get("HM", true)) {
					String[] parts = l.split("\\|");
					items.add(parts[0]);
				}

				Counter<String> c1 = Generics.newCounter();
				for (String item : items) {
					c1.incrementCount(item, 1);
				}
				c.incrementAll(c1);

			}
			reader.close();
		}

		{
			/**
			 * 
			 * AN ANNOTATION DA DATE OF ENTRY
			 * 
			 * DQ DATE QUALIFIER ESTABLISHED
			 * 
			 * HN HISTORY NOTE
			 * 
			 * MR MAJOR REVISION DATE
			 * 
			 * MS MESH SCOPE NOTE
			 * 
			 * OL ONLINE NOTE
			 * 
			 * QA TOPICAL QUALIFIER ABBREVIATION
			 * 
			 * QE QUALIFIER ENTRY VERSION
			 * 
			 * QS QUALIFIER SORT VERSION
			 * 
			 * QT QUALIFIER TYPE
			 * 
			 * QX QUALIFIER CROSS REFERENCE
			 * 
			 * RECTYPE RECORD TYPE [= RY in ELHILL MESH]
			 * 
			 * SH SUBHEADING
			 * 
			 * TN TREE NODE ALLOWED
			 * 
			 * UI UNIQUE IDENTIFIER
			 */
			TextFileReader reader = new TextFileReader(MIRPath.MESH_COL_RAW_QUALIFIER_FILE);

			while (reader.hasNext()) {
				List<String> lines = reader.nextLines();
				ListMap<String, String> lm = Generics.newListMap();

				for (int i = 1; i < lines.size();) {
					String l1 = lines.get(i);
					int end = i + 1;

					for (int j = i + 1; j < lines.size(); j++) {
						String l2 = lines.get(j);
						if (l2.contains(sep)) {
							end = j;
							break;
						}
					}

					String s = StrUtils.join(" ", lines, i, end);
					String[] two = s.split(sep);
					String attr = two[0];
					String value = two[1];
					lm.put(attr, value);

					i = end;
				}

				if (!lm.containsKey("NM")) {
					continue;
				}

				String name = lm.get("NM", true).get(0);

				List<String> items = Generics.newArrayList();
				items.add(name);

				for (String l : lm.get("QX", true)) {
					String[] parts = l.split("\\|");
					items.add(parts[0]);
				}

				Counter<String> c1 = Generics.newCounter();
				for (String item : items) {
					c1.incrementCount(item, 1);
				}
				c.incrementAll(c1);

			}
			reader.close();
		}

		// for(String item : c.keySet()){
		// if(item.contains("(") && item.contains(")")){
		// System.out.println(item);
		// }
		// }

		Counter<String> c2 = Generics.newCounter();

		for (Entry<String, Double> e : c.entrySet()) {
			String phrs = e.getKey();
			double cnt = e.getValue();

			if (phrs.startsWith("*")) {
				phrs = phrs.substring(1, phrs.length());
			}
			c2.setCount(phrs, cnt);
		}

		FileUtils.writeStringCounterAsText(MIRPath.DATA_DIR + "phrs/mesh_phrs.txt", c2);

	}

	public void getScopusKeywords() throws Exception {
		List<File> files = FileUtils.getFilesUnder(MIRPath.DATA_DIR + "scopus/col/line/");
		Counter<String> c = Generics.newCounter();
		for (File file : files) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] ps = line.split("\t");
				ps = StrUtils.unwrap(ps);
				if (ps.length == 4) {
					for (String kwd : ps[3].split(StrUtils.LINE_REP)) {
						kwd = StrUtils.normalizeSpaces(kwd);
						if (kwd.length() > 0) {
							c.incrementCount(kwd, 1);
						}
					}
				}
			}
		}
		FileUtils.writeStringCounterAsText(MIRPath.DATA_DIR + "phrs/scopus_kwd.txt", c);
	}

	/**
	 * 
	 * https://confluence.ihtsdotools.org/display/DOCRELFMT/3.2.2.+Description+File+Specification
	 * 
	 * @throws Exception
	 */
	public void getSnomedConcepts() throws Exception {
		String inFile = MIRPath.SNOMED_DIR
				+ "SnomedCT_USEditionRF2_Production_20170301T120000/Snapshot/Terminology/sct2_Description_Snapshot-en_US1000124_20170301.txt";

		String outFile = MIRPath.SNOMED_DIR + "cpts.txt";

		TextFileReader reader = new TextFileReader(inFile);

		Map<String, Boolean> activeMap = Generics.newHashMap();
		ListMap<String, String> lm = Generics.newListMap();

		String regex = "\\([^\\(\\)]+\\)";
		Pattern p = Pattern.compile(regex);

		while (reader.hasNext()) {
			if (reader.getLineCnt() == 1) {
				continue;
			}
			String line = reader.next();
			String[] ps = line.split("\t");

			int loc = 0;
			String id = ps[loc++];
			String effectiveTime = ps[loc++];
			boolean active = ps[loc++].equals("1") ? true : false;
			String moduled = ps[loc++];
			String cptId = ps[loc++];
			String langCode = ps[loc++];
			String typeId = ps[loc++];
			String term = ps[loc++];
			String caseSignificanceId = ps[loc++];

			term = term.replace("( &", "(&");
			term = term.replace("[D]", "");
			term = term.replace("[V]", "");
			term = term.replaceAll("(,) NOS", "");

			lm.put(cptId, typeId + "\t" + term);
			activeMap.put(cptId, active);
		}
		reader.close();

		ListMapMap<String, String, String> lmm = Generics.newListMapMap(ListType.LINKED_LIST);

		for (String cptId : lm.keySet()) {
			List<String> items = lm.get(cptId);
			Map<String, String> m = Generics.newHashMap(items.size());

			for (String s : items) {
				String[] ps = s.split("\t");
				m.put(ps[0], ps[1]);
			}

			String semType = "none";
			String name = m.remove("900000000000003001");
			List<String> variants = Generics.newArrayList(m.values());

			List<Integer> idxs1 = Generics.newArrayList(10);
			List<Integer> idxs2 = Generics.newArrayList(10);

			for (int i = 0; i < name.length(); i++) {
				char ch = name.charAt(i);

				if (ch == '(') {
					idxs1.add(i);
				} else if (ch == ')') {
					idxs2.add(i);
				}
			}

			if (idxs1.size() == idxs2.size()) {
				if (idxs1.size() == 1) {
					int start = idxs1.get(idxs1.size() - 1);
					int end = idxs2.get(idxs2.size() - 1);
					if (end == name.length() - 1 && name.charAt(start + 1) != '&') {
						semType = name.substring(start + 1, end);
						name = name.substring(0, start).trim();
					}
				} else if (idxs1.size() > 1) {
					int idx1 = name.indexOf(variants.get(0));
					if (idx1 >= 0) {
						int idx2 = idx1 + variants.get(0).length();

						String s = name.substring(idx2, name.length()).trim();

						if (s.startsWith("(") && s.endsWith(")")) {
							semType = s.substring(1, s.length() - 1);
							name = name.substring(0, idx2).trim();
						}
					}
				}
			}

			Set<String> set = Generics.newHashSet();
			set.add(name);
			set.addAll(variants);

			items.clear();

			if (set.size() == 1) {
				items.addAll(set);
			} else {
				items.add(name);
				items.addAll(variants);
			}

			lmm.put(semType, cptId, items);
		}

		Counter<String> c = Generics.newCounter();

		List<String> res = Generics.newLinkedList();

		List<String> semTypes = Generics.newArrayList(lmm.keySet());
		Collections.sort(semTypes);

		for (String semType : semTypes) {
			ListMap<String, String> lm2 = lmm.get(semType);

			List<String> cptIds = Generics.newArrayList(lm2.keySet());
			Collections.sort(cptIds);

			for (String cptId : cptIds) {
				List<String> l = lm2.get(cptId);
				res.add(semType + "\t" + cptId + "\t" + StrUtils.join("\t", l));
				c.incrementCount(semType, l.size());
			}
		}

		System.out.println(c.toStringSortedByValues(true, true, c.size(), "\t"));

		FileUtils.writeStringCollectionAsText(outFile, res);
	}

	public void getSnomedPhrases() throws Exception {
		String inFile = MIRPath.SNOMED_DIR + "cpts.txt";
		String outFile = MIRPath.DATA_DIR + "phrs/snomed_phrss.txt";

		Set<String> validType = Generics.newHashSet();
		validType.add("disorder");
		validType.add("procedure");
		validType.add("finding");
		validType.add("organism");
		validType.add("body structure");
		validType.add("tumor staging");
		validType.add("cell");
		validType.add("organism");
		validType.add("substance");
		validType.add("morphologic abnormality");

		Counter<String> c = Generics.newCounter();

		TextFileReader reader = new TextFileReader(inFile);

		while (reader.hasNext()) {
			if (reader.getLineCnt() == 1) {
				continue;
			}

			String line = reader.next();
			String[] ps = line.split("\t");

			String semType = ps[0];

			if (!validType.contains(semType)) {
				continue;
			}

			for (int i = 2; i < ps.length; i++) {
				String cpt = ps[i];
				c.incrementCount(cpt, 1);
			}

		}
		reader.close();

		FileUtils.writeStringCounterAsText(outFile, c, true);
	}

	public void getTrecCdsKeywords() throws Exception {
		List<File> files = FileUtils.getFilesUnder(MIRPath.TREC_CDS_2016_COL_LINE_DIR);
		Counter<String> c = Generics.newCounter();
		for (File file : files) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] ps = line.split("\t");
				ps = StrUtils.unwrap(ps);
				for (String kwd : ps[4].split(StrUtils.LINE_REP)) {
					kwd = StrUtils.normalizeSpaces(kwd);
					if (kwd.length() > 0) {
						c.incrementCount(kwd, 1);
					}
				}
			}
		}
		FileUtils.writeStringCounterAsText(MIRPath.DATA_DIR + "phrs/cds_kwd.txt", c);
	}

	public void getTrecPmClinicalKeywords() throws Exception {
		List<File> files = FileUtils.getFilesUnder(MIRPath.TREC_PM_2017_COL_CLINICAL_LINE_DIR);
		Counter<String> c = Generics.newCounter();
		for (File file : files) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] ps = line.split("\t");
				ps = StrUtils.unwrap(ps);
				for (String kwd : ps[4].split(StrUtils.LINE_REP)) {
					kwd = StrUtils.normalizeSpaces(kwd);
					if (kwd.length() > 0) {
						c.incrementCount(kwd, 1);
					}
				}
			}
		}
		FileUtils.writeStringCounterAsText(MIRPath.DATA_DIR + "phrs/pmct_kwd.txt", c);
	}

	public void getWikiPhrases() throws Exception {
		CounterMap<String, String> cm = Generics.newCounterMap();

		String reg = "\\([^\\(\\)]+\\)";

		Pattern p = Pattern.compile(reg);

		for (File file : FileUtils.getFilesUnder(MIRPath.WIKI_COL_LINE_DIR)) {
			for (String line : FileUtils.readLinesFromText(file)) {
				String[] vals = line.split("\t");

				if (vals.length != 5) {
					continue;
				}

				vals = StrUtils.unwrap(vals);

				String title = vals[2];
				String phrss = vals[4];

				Matcher m = p.matcher(title);
				StringBuffer sb = new StringBuffer();

				if (m.find()) {
					String g = m.group();
					// System.out.println(g);
					// int s = m.start();
					m.appendReplacement(sb, "");
				}
				m.appendTail(sb);

				title = sb.toString().trim();

				if (title.startsWith("List of")) {
					title = title.substring("List of".length() + 1);
				}

				String t = title;
				// t = sn.normalize(t);

				if (t.split(" ").length > 1 && !t.contains("?")) {
					cm.incrementCount(t, "wkt", 1);
				}

				for (String phrs : phrss.split("\\|")) {
					// phrs = sn.normalize(phrs);
					if (phrs.split(" ").length > 1 && !phrs.contains("?")) {
						cm.incrementCount(phrs, "wkl", 1);
					}
				}
			}
		}

		FileUtils.writeStringCounterMapAsText(MIRPath.DATA_DIR + "phrs/wiki_phrs.txt", cm);
	}

	public void matchPhrasesToKeywords() throws Exception {
		Counter<String> c = FileUtils.readStringCounterFromText(KPPath.KYP_DIR + "phrs_3p_kwds.txt.gz");

		List<String> lines = FileUtils.readLinesFromText(KPPath.KYP_DIR + "phrs_3p.txt.gz");

		Counter<String> cc = Generics.newCounter();

		for (int i = 0; i < lines.size(); i++) {
			List<String> parts = Generics.newArrayList();

			for (String s : lines.get(i).split("\t")) {
				parts.add(s);
			}

			int cnt = (int) c.getCount(parts.get(0));

			if (cnt > 0) {
				cc.incrementCount("Phrs+Kwd", 1);
			}

			cc.incrementCount("Phrs", 1);

			parts.add(cnt + "");

			lines.set(i, StrUtils.join("\t", parts));
		}

		System.out.println(cc);

		FileUtils.writeStringCollectionAsText(KPPath.KYP_DIR + "phrs_3p_label.txt.gz", lines);
	}

	public void mergeKeywords() throws Exception {
		String[] fileNames = { "cds_kwd.txt", "mesh_phrs.txt", "pmct_kwd.txt", "scopus_kwd.txt", "snomed_phrs.txt", "wiki_phrs.txt" };
		String[] names = { "cds", "mes", "pm", "sco", "sno", "wkt" };

		CounterMap<String, String> cm = Generics.newCounterMap(2000000);

		for (int i = 0; i < fileNames.length; i++) {
			String fileName = MIRPath.DATA_DIR + "phrs/" + fileNames[i];
			String name = names[i];

			if (fileName.contains("wiki")) {
				CounterMap<String, String> cm2 = FileUtils.readStringCounterMapFromText(fileName);
				for (String phrs : cm2.keySet()) {
					Counter<String> c = cm2.getCounter(phrs);
					cm.incrementCount(phrs.toLowerCase(), name, c.totalCount());
				}
			} else {
				Counter<String> c = FileUtils.readStringCounterFromText(fileName);
				for (Entry<String, Double> e : c.entrySet()) {
					String phrs = e.getKey();
					double cnt = e.getValue();
					cm.incrementCount(phrs.toLowerCase(), name, cnt);
				}
			}
		}

		Counter<String> c = Generics.newCounter();

		for (String phrs : cm.keySet()) {
			c.setCount(phrs, cm.getCounter(phrs).size());
		}

		List<String> res = Generics.newArrayList(cm.size());

		for (String phrs : c.getSortedKeys()) {
			int len = (int) c.getCount(phrs);
			Counter<String> c2 = cm.removeKey(phrs);
			List<String> rs = Generics.newArrayList(c2.keySet());
			Collections.sort(rs);
			res.add(phrs + "\t" + (int) c2.totalCount() + "\t" + len + "\t" + StrUtils.join("\t", rs));
		}

		FileUtils.writeStringCollectionAsText(MIRPath.DATA_DIR + "phrs/phrs_merged.txt", res);
	}

}
