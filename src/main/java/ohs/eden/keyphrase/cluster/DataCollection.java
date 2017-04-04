package ohs.eden.keyphrase.cluster;

import java.io.File;
import java.util.List;

import ohs.io.FileUtils;
import ohs.types.generic.ListMap;
import ohs.utils.Generics;

public class DataCollection {

	private String inDir;

	private ListMap<String, File> typeToFiles;

	private int num_files;

	public DataCollection(String inDir) {
		this.inDir = inDir;

		setup();
	}

	public List<String> getTypes() {
		return Generics.newArrayList(typeToFiles.keySet());
	}

	public int size(String type) {
		return typeToFiles.get(type).size();
	}

	public int totalSize() {
		return typeToFiles.sizeOfEntries();
	}

	public List<String> getDocs(String type, int i) throws Exception {
		return FileUtils.readLinesFromText(typeToFiles.get(type).get(i).getPath());
	}

	public File getFile(String type, int i) throws Exception {
		return typeToFiles.get(type).get(i);
	}

	public List<File> getFiles(String type) throws Exception {
		return typeToFiles.get(type);
	}

	private void setup() {
		typeToFiles = Generics.newListMap();

		for (File dir : new File(inDir).listFiles()) {
			String dirName = dir.getName();

			for (File file : dir.listFiles()) {
				typeToFiles.put(dirName, file);
				num_files++;
			}
		}
	}

}
