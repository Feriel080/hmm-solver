import { spawn } from 'child_process';

function executeCommand(command: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args);
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('error', (error) => {
      reject(error);
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Process exited with code ${code}: ${stderr}`));
      } else {
        resolve(stdout);
      }
    });

    // Set a timeout
    setTimeout(() => {
      child.kill();
      reject(new Error('Python process timeout'));
    }, 30000);
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();

    // Execute Python executor with JSON argument
    const pythonScript = 'lib/hmm_executor.py';
    const jsonArg = JSON.stringify(body);

    const stdout = await executeCommand('py', [pythonScript, jsonArg]);

    const result = JSON.parse(stdout);
    return Response.json(result);
  } catch (error: any) {
    console.error('API error:', error);
    return Response.json(
      { error: 'Failed to process request', details: error.message },
      { status: 500 }
    );
  }
}
