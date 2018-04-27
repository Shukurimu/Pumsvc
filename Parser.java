import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.HashMap;
import java.util.ArrayList;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.Charset;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

public class Parser {
	
	public static void main(String[] args) throws Exception {
		if (args.length < 1) {
			System.out.println("argument: InputFileName");
			return;
		}
		final DateTimeFormatter dp = DateTimeFormatter.ofPattern("yyyyMMdd");
		final HashMap<String, Integer> statistic = new HashMap<>();
		final Pattern entryPattern = Pattern.compile("\"(.*?)\"");
		final PrintWriter pw = new PrintWriter(args[0].replaceAll("\\..*$", ".txt"), "utf-8");
		Files.lines(Paths.get(args[0]), Charset.forName("big5-hkscs")).skip(1L).forEach(s -> {
			ArrayList<String> entry = new ArrayList<String>(8);
			Matcher entryMatcher = entryPattern.matcher(s.replaceAll(" ", ""));
			while (entryMatcher.find())
				entry.add(entryMatcher.group(1).replaceAll(",", ""));
			String stock = entry.get(0) + " " + entry.get(2);
			statistic.put(stock, statistic.getOrDefault(stock, 0) + 1);
			final LocalDate date = LocalDate.parse(entry.get(1), dp);
			entry.set(1, Integer.toString(date.getYear()));
			entry.set(2, Integer.toString(date.getMonthValue() % 12));
			entry.add(3, Integer.toString(date.getDayOfWeek().getValue() % 7));
			pw.println(String.join(" ", entry));
		});
		pw.close();
		PrintWriter st = new PrintWriter(args[0].replaceAll("\\..*$", "_stat.txt"), "utf-8");
		statistic.forEach((k, v) -> st.println(k + " " + v));
		st.close();
		return;
	}
	
}
