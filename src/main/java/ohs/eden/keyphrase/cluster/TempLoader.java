package ohs.eden.keyphrase.cluster;

import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.List;

import ohs.io.TextFileReader;
import ohs.types.generic.Counter;
import ohs.utils.Generics;

public class TempLoader {

	public static void main(String[] args) throws Exception {
		System.out.println("process begins.");

		test();

		TempLoader loader = new TempLoader();
		// loader.open();
		//
		// List<String> lines = FileUtils.readLinesFromText("../../data/terms.txt");
		// loader.loadKeywordsMap(lines);
		//
		// loader.close();
		System.out.println("process ends.");
	}

	public static void test() {

		Counter<String> c1 = Generics.newCounter();

		{
			TextFileReader reader = new TextFileReader("../../data/terms/term_trans.csv");
			while (reader.hasNext()) {
				String line = reader.next();

				if (reader.getLineCnt() < 50) {
					System.out.println(line);
				}
				c1.incrementCount(line, 1);
			}
			reader.close();
		}

		System.out.println(c1.toString());

		Counter<String> c2 = Generics.newCounter();

		{
			TextFileReader reader = new TextFileReader("../../data/terms/terms-2.txt");
			while (reader.hasNext()) {

				if (reader.getLineCnt() == 1) {
					continue;
				}

				String line = reader.next();

				// if (reader.getLineCnt() > 50) {
				// break;
				// }

				c2.incrementCount(line, 1);
			}
			reader.close();
		}

		System.out.println(c2.toString());

		System.out.printf("cnt1\t%d\n", (int) c1.totalCount());
		System.out.printf("cnt2\t%d\n", (int) c2.totalCount());

		int cnt3 = 0;

		for (String key : c2.keySet()) {
			if (c1.containsKey(key)) {
				cnt3 += c2.getCount(key);
			}
		}

		System.out.printf("cnt3\t%d\n", cnt3);

	}

	private Connection con;

	private int batch_size = 10000;

	public TempLoader() {

	}

	public void close() throws Exception {
		con.close();
	}

	public void loadKeywordsMap(List<String> lines) throws Exception {
		String sql = "update term_trans set trans=? where term_id=? and seq='1' and trans_seq='3' and reg_div='S' and trans_div='2'";

		List<String> tmpLines = Generics.newArrayList();
		int batch_cnt = 0;

		for (int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			if (tmpLines.size() == batch_size || i == lines.size() - 1) {
				System.out.printf("%dth: %d\n", ++batch_cnt, tmpLines.size());
				PreparedStatement pstmt = con.prepareStatement(sql);

				for (String s : tmpLines) {
					String[] parts = s.split("\t");
					String f = parts[1];
					int id = Integer.parseInt(parts[0]);

					pstmt.setString(1, f);
					pstmt.setInt(2, id);
					pstmt.addBatch();
				}

				pstmt.executeBatch();
				con.commit();
				pstmt.close();

				tmpLines = Generics.newArrayList();
			} else {
				tmpLines.add(line);
			}
		}

	}

	public void open() throws Exception {
		Class.forName("com.mysql.jdbc.Driver");
		String url = String.format("jdbc:mysql://203.250.196.45:3306/knoba_data");
		String id = "knoba";
		String pwd = "knoba123#";

		con = DriverManager.getConnection(url, id, pwd);
		con.setAutoCommit(false);
	}

	private void printMetaData(String tableName) throws Exception {
		DatabaseMetaData meta = con.getMetaData();
		ResultSet resultSet = meta.getColumns(null, null, tableName, null);
		while (resultSet.next()) {
			String name = resultSet.getString("COLUMN_NAME");
			String type = resultSet.getString("TYPE_NAME");
			int size = resultSet.getInt("COLUMN_SIZE");

			System.out.println("Column name: [" + name + "]; type: [" + type + "]; size: [" + size + "]");
		}
	}

}
