package ohs.corpus.dump;

import java.util.List;

import ohs.io.TextFileReader;
import ohs.io.TextFileWriter;
import ohs.ir.medical.general.MIRPath;
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
public class MeSHDumper {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		MeSHDumper h = new MeSHDumper();
		h.dump();
		// h.extract();
		// h.extractQualifierTree();

		System.out.println("process ends.");
	}

	public void dump() throws Exception {

		/***
		 * 
		 * 
		 * AN ANNOTATION
		 * 
		 * AQ ALLOWABLE TOPICAL QUALIFIERS
		 * 
		 * CATSH CATALOGING SUBHEADINGS LIST DATA_NAME
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
		 * N1 CAS TYPE 1 DATA_NAME
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

}
