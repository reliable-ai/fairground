import fs from "node:fs";
import path from "node:path";

export default {
  watch: ["./collection_*.json"],

  load(watchedFiles) {
    // If we have watched files (in development), use the first one
    if (watchedFiles && watchedFiles.length > 0) {
      // Iterate over watched files, load them and create an object with the file name as key
      // and the parsed content as value
      const collections = {};
      watchedFiles.forEach((file) => {
        const fileName = path.basename(file).replace(".json", "");
        const content = fs.readFileSync(file, "utf-8");
        collections[fileName] = JSON.parse(content);
      });
      return collections;
    }
    return {}; // Return empty object if no files are watched
  },
};
