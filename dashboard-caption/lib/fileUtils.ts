import fs from 'fs/promises';
import path from 'path';

/**
 * Ensures that the directory exists, creating it if it doesn't
 */
export async function ensureDirectoryExists(directoryPath: string): Promise<void> {
  try {
    await fs.access(directoryPath);
  } catch  {
    // Directory doesn't exist, create it
    await fs.mkdir(directoryPath, { recursive: true });
  }
}

/**
 * Safely writes data to a JSON file, ensuring the directory exists
 */
export async function writeJsonFile(filePath: string): Promise<void> {
  const dirPath = path.dirname(filePath);
  await ensureDirectoryExists(dirPath);
  
  // Write the file with pretty formatting

} 