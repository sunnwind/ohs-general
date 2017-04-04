package ohs.bioasq;

import java.util.List;
import java.util.Set;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ohs.corpus.search.app.DocumentSearcher;
import ohs.corpus.search.index.WordFilter;
import ohs.corpus.type.DocumentCollection;
import ohs.corpus.type.SimpleStringNormalizer;
import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.io.TextFileWriter;
import ohs.ir.medical.general.MIRPath;
import ohs.ir.medical.general.NLPUtils;
import ohs.types.generic.Counter;
import ohs.types.generic.CounterMap;
import ohs.types.generic.ListMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

/**
 * 
 * https://www.nlm.nih.gov/mesh/introduction.html
 * 
 * https://www.nlm.nih.gov/mesh/download_mesh.html
 * 
 * @author ohs
 */
public class MeSHHandler {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		MeSHHandler h = new MeSHHandler();
		// h.dump();
		h.extract();
		// h.extractQualifierTree();

		System.out.println("process ends.");
	}

	public void extractQualifierTree() throws Exception {
		String inFileName = MIRPath.MESH_COL_RAW_DIR + "2017MeshTree_qualifier_raw.txt";

		List<String> lines = FileUtils.readLinesFromText(inFileName);

		String regex = "\\([^\\(\\)]+\\)";

		Pattern p = Pattern.compile(regex);

		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			Matcher m = p.matcher(line);

			m.find();
			String g = m.group();
			g = g.substring(1, g.length() - 1);

			String label = line.substring(0, m.start());

			System.out.println(g + "\t" + label);

			lines.set(i, g + "\t" + label);
		}

		FileUtils.writeStringCollection(inFileName.replace("_raw", ""), lines);
	}

	public void dump() throws Exception {

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

		TextFileReader reader = new TextFileReader(MIRPath.MESH_COL_RAW_DESCRIPTOR_FILE);
		TextFileWriter writer = new TextFileWriter(MIRPath.MESH_COL_LINE_DIR + "00000.txt.gz");
		String sep = " = ";

		CounterMap<String, String> cm = Generics.newCounterMap();

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

			List<String> terms = Generics.newArrayList();
			terms.add(name);

			for (String l : lm.get("ENTRY", true)) {
				String[] parts = l.split("\\|");
				terms.add(parts[0]);
			}

			for (String l : lm.get("PRINT ENTRY", true)) {
				String[] parts = l.split("\\|");
				terms.add(parts[0]);
			}

			if (lm.containsKey("MS")) {
				terms.add(lm.get("MS").get(0));
			}

			if (lm.containsKey("MH") && lm.containsKey("MN")) {

			} else {
				continue;
			}

			// String treeNumber = lm.get("MN", true).get(0);

			// name = StrUtils.join(" ", NLPUtils.tokenize(name.toLowerCase()));

			String[] res = new String[] { name, StrUtils.join(StrUtils.LINE_REP, terms) };
			res = StrUtils.wrap(res);

			writer.write(StrUtils.join("\t", res) + "\n");

		}
		reader.close();
		writer.close();
	}

	private String normalize(String s) {
		String[] ps = s.split(", ");
		if (ps.length > 1) {
			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < ps.length; i++) {
				String p = ps[ps.length - i - 1];
				sb.append(p);
				if (i != ps.length - 1) {
					sb.append(" ");
				}
				s = sb.toString();
			}
		}

		s = s.replace("*", "");

		return s;
	}

	public void extract() throws Exception {

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

		DocumentCollection dc = new DocumentCollection(MIRPath.TREC_CDS_2016_COL_DC_DIR);
		Set<String> stopwords = FileUtils.readStringSetFromText(MIRPath.STOPWORD_INQUERY_FILE);
		WordFilter filter = new WordFilter(dc.getVocab(), stopwords);

		SimpleStringNormalizer sn = new SimpleStringNormalizer(true);

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
					item = normalize(item);
					item = sn.normalize(item);
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
					item = normalize(item);
					item = sn.normalize(item);
					c1.incrementCount(item.toLowerCase(), 1);
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
					item = normalize(item);
					item = sn.normalize(item);
					c1.incrementCount(item.toLowerCase(), 1);
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

		FileUtils.writeStringCounterAsText(MIRPath.MESH_DIR + "phrss.txt", c);

	}

}
