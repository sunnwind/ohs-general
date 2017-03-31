package ohs.bioasq;

import java.io.File;
import java.io.FileInputStream;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonObject;
import javax.json.JsonReader;

import ohs.io.FileUtils;
import ohs.ir.medical.general.MIRPath;
import ohs.types.generic.ListMap;
import ohs.utils.Generics;
import ohs.utils.StrUtils;

public class TestArticleReader {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		mergeUnder(MIRPath.BIOASQ_DIR + "test/task4a/", MIRPath.BIOASQ_DIR + "test/task4a_all.txt");

		Map<String, String> map = Generics.newHashMap();

		for (String line : FileUtils.readLinesFromText(MIRPath.BIOASQ_DIR + "test/task4a_pmid_mesh.txt")) {
			String[] parts = line.split("\t");
			String pmid = parts[0];

			List<String> meshes = Generics.newArrayList();

			for (int i = 1; i < parts.length; i++) {
				String mesh = parts[i];
				int idx = mesh.indexOf("/");
				if (idx > -1) {
					mesh = mesh.substring(0, idx);
				}
				mesh = mesh.replace("*", "").trim();
				meshes.add(mesh);
			}

			map.put(pmid, StrUtils.join("|", meshes));
		}

		List<String> lines = Generics.newArrayList();

		for (String line : FileUtils.readLinesFromText(MIRPath.BIOASQ_DIR + "test/task4a_all.txt")) {
			String[] parts = line.split("\t");
			String pmid = parts[0];
			String journal = parts[1];
			String title = parts[2];
			String abs = parts[3];

			String mesh = map.get(pmid);

			if (mesh == null) {
				continue;
			}

			lines.add(line + "\t" + mesh);
		}

		FileUtils.writeStringCollection(MIRPath.BIOASQ_DIR + "test/task4a_all_mesh.txt", lines);

		System.out.println("process ends.");
	}

	public static void mergeUnder(String dirName, String outFileName) throws Exception {
		ListMap<String, String> l = read(FileUtils.getFilesUnder(dirName));

		List<String> pmids = Generics.newArrayList(l.keySet());
		Collections.sort(pmids);

		List<String> outs = Generics.newArrayList();

		for (String pmid : pmids) {
			outs.add(pmid + "\t" + StrUtils.join("\t", l.get(pmid)));
		}

		FileUtils.writeStringCollection(outFileName, outs);
	}

	public static ListMap<String, String> read(List<File> files) throws Exception {
		ListMap<String, String> ret = Generics.newListMap();

		for (File file : files) {
			ListMap<String, String> l = read(file.getPath());

			for (String pmid : l.keySet()) {
				if (ret.containsKey(pmid)) {
					System.out.printf("%s, %s, %s\n", file.getName(), pmid, l.get(pmid));
				} else {
					ret.put(pmid, l.get(pmid));
				}
			}
		}

		return ret;

	}

	/**
	 * 
	 * http://www.journaldev.com/2315/java-json-example
	 * 
	 * @param fileName
	 * @throws Exception
	 */
	public static ListMap<String, String> read(String fileName) throws Exception {
		ListMap<String, String> ret = Generics.newListMap();

		JsonReader jr = Json.createReader(new FileInputStream(fileName));
		JsonObject jo = jr.readObject();

		JsonArray docs = jo.getJsonArray("documents");

		for (int i = 0; i < docs.size(); i++) {
			JsonObject doc = docs.getJsonObject(i);

			String pmid = doc.get("pmid").toString();
			String title = doc.get("title").toString();
			String abs = doc.get("abstractText").toString();
			String journal = doc.getString("journal").toString();

			if (pmid.startsWith("\"")) {
				pmid = StrUtils.unwrap(pmid);
			}

			if (title.startsWith("\"")) {
				title = StrUtils.unwrap(title);
			}

			if (abs.startsWith("\"")) {
				abs = StrUtils.unwrap(abs);
			}

			if (journal.startsWith("\"")) {
				journal = StrUtils.unwrap(journal);
			}

			ret.put(pmid, journal);
			ret.put(pmid, title);
			ret.put(pmid, abs);
		}
		return ret;
	}

	public static void read2() {

	}

	public static void readMeshMap(String fileName) throws Exception {

		for (String line : FileUtils.readLinesFromText(fileName)) {
			String[] parts = line.split("\t");
			String pmid = parts[0];

			List<String> meshes = Generics.newArrayList();

			for (int i = 1; i < parts.length; i++) {
				String mesh = parts[i];
				int idx = mesh.indexOf("/");
				mesh = mesh.substring(0, idx);
				mesh = mesh.replace("*", "");
			}
		}
	}

}
