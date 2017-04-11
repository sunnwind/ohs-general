package ohs.bioasq;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

import ohs.corpus.type.SimpleStringNormalizer;
import ohs.io.FileUtils;
import ohs.io.TextFileReader;
import ohs.ir.medical.general.MIRPath;
import ohs.types.generic.Counter;
import ohs.types.generic.ListMap;
import ohs.types.generic.ListMapMap;
import ohs.utils.Generics;
import ohs.utils.Generics.ListType;
import ohs.utils.StrUtils;

/**
 * 
 * https://confluence.ihtsdotools.org/display/DOCRELFMT/3.2.2.+Description+File+Specification
 * 
 * @author ohs
 */
public class SnomedCTHandler {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		SnomedCTHandler h = new SnomedCTHandler();
		// h.getConcepts();
		h.getPhrases();

		System.out.println("process ends.");
	}

	public void getConcepts() throws Exception {
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

	public void getPhrases() throws Exception {
		String inFile = MIRPath.SNOMED_DIR + "cpts.txt";
		String outFile = MIRPath.SNOMED_DIR + "phrss.txt";

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

		SimpleStringNormalizer sn = new SimpleStringNormalizer(true);

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
				String phrs = sn.normalize(cpt);
				c.incrementCount(phrs, 1);
			}

		}
		reader.close();

		FileUtils.writeStringCounterAsText(outFile, c, true);
	}

}
