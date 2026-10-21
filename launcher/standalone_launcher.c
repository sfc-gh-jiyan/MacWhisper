/*
 * Standalone MacWhisper launcher — the app runs entirely from inside its bundle.
 *
 * Bundle layout (relative to Contents/):
 *   MacOS/MacWhisper              this launcher
 *   Resources/python/             relocatable CPython (app runtime, mlx-whisper)
 *   Resources/python-qwen/        relocatable CPython (Qwen3 sidecar, newer mlx)
 *   Resources/app/                app sources (+ SystemAudioHelper)
 *
 * The launcher embeds CPython in-process (PEP 587). It never re-execs: exec
 * gives the process a new pid version and macOS can no longer resolve the app
 * (no menu bar icon, permission mismatch).
 *
 * Environment set for the app (existing values win, so tests can override):
 *   MACWHISPER_DATA_DIR     ~/Library/Application Support/MacWhisper
 *   MACWHISPER_QWEN_PYTHON  Resources/python-qwen/bin/python3
 *   SSL_CERT_FILE           certifi bundle inside Resources/python
 *
 * Usage:
 *   MacWhisper               run the app (stdout/stderr -> <data>/logs/macwhisper.log)
 *   MacWhisper --self-test   import every runtime dependency, print a report, exit
 *   MW_SELF_TEST=<file>      same self-test when started through LaunchServices
 */
#include <Python.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define PY_VER "3.13"

static const char *SELF_TEST_CODE =
    "import os, sys, subprocess\n"
    "print('executable=' + sys.executable)\n"
    "print('prefix=' + sys.prefix)\n"
    "print('cwd=' + os.getcwd())\n"
    "print('data_dir=' + os.environ.get('MACWHISPER_DATA_DIR', ''))\n"
    "print('MallocNanoZone=' + os.environ.get('MallocNanoZone', ''))\n"
    "print('SSL_CERT_FILE=' + os.environ.get('SSL_CERT_FILE', ''))\n"
    "outside = [p for p in sys.path if p and not p.startswith(os.path.dirname(os.path.dirname(sys.prefix)))]\n"
    "print('sys_path_outside_bundle=' + repr(outside))\n"
    "import rumps, objc, mlx_whisper, torch, sounddevice, pynput, numpy, opencc, pyperclip\n"
    "import paths, audio_capture\n"
    "print('helper=' + str(audio_capture._find_system_audio_helper()))\n"
    "print('imports=ok')\n"
    "q = os.environ['MACWHISPER_QWEN_PYTHON']\n"
    "r = subprocess.run([q, '-I', '-c', 'import mlx_audio, mlx.core as mx; print(mx.__version__)'],\n"
    "                   capture_output=True, text=True, timeout=120)\n"
    "print('qwen_python=' + q)\n"
    "print('qwen_imports=' + ('ok mlx ' + r.stdout.strip() if r.returncode == 0 else 'FAIL ' + r.stderr[-400:]))\n";

static char contents_dir[PATH_MAX];  /* .../MacWhisper.app/Contents */

static void die(const char *what) {
    fprintf(stderr, "[MacWhisper launcher] %s: %s\n", what, strerror(errno));
    exit(1);
}

static void find_contents_dir(void) {
    char exe[PATH_MAX], real[PATH_MAX];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0 || !realpath(exe, real))
        die("cannot resolve executable path");
    /* real = Contents/MacOS/MacWhisper -> Contents */
    char *macos = dirname(real);
    strlcpy(contents_dir, dirname(macos), sizeof(contents_dir));
}

static void bundle_path(char *out, size_t n, const char *rel) {
    snprintf(out, n, "%s/%s", contents_dir, rel);
}

static void setenv_default(const char *k, const char *v) {
    const char *cur = getenv(k);
    if (!cur || !cur[0])
        setenv(k, v, 1);
}

static void mkdirs(const char *path) {
    char tmp[PATH_MAX];
    strlcpy(tmp, path, sizeof(tmp));
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static void warn_if_malloc_env_missing(void) {
    const char *v = getenv("MallocNanoZone");
    if (!v || strcmp(v, "0") != 0)
        fprintf(stderr, "[MacWhisper launcher] MallocNanoZone=0 not set "
                        "(start the app via LaunchServices, not the binary)\n");
}

static void redirect_output(const char *file, int truncate) {
    int fd = open(file, O_WRONLY | O_CREAT | (truncate ? O_TRUNC : O_APPEND), 0644);
    if (fd < 0)
        return;  /* keep the default output rather than fail to start */
    dup2(fd, STDOUT_FILENO);
    dup2(fd, STDERR_FILENO);
    close(fd);
}

static int fail_status(PyConfig *config, PyStatus status) {
    PyConfig_Clear(config);
    if (PyStatus_IsExit(status))
        return status.exitcode;
    Py_ExitStatusException(status);
    return 1;
}

int main(int argc, char **argv) {
    const char *self_test_file = getenv("MW_SELF_TEST");
    int self_test = (argc > 1 && strcmp(argv[1], "--self-test") == 0) ||
                    (self_test_file && self_test_file[0]);

    find_contents_dir();
    char py_home[PATH_MAX], py_exe[PATH_MAX], app_dir[PATH_MAX], script[PATH_MAX];
    char qwen_py[PATH_MAX], cacert[PATH_MAX];
    bundle_path(py_home, sizeof(py_home), "Resources/python");
    bundle_path(py_exe, sizeof(py_exe), "Resources/python/bin/python3");
    bundle_path(app_dir, sizeof(app_dir), "Resources/app");
    bundle_path(qwen_py, sizeof(qwen_py), "Resources/python-qwen/bin/python3");
    bundle_path(cacert, sizeof(cacert),
                "Resources/python/lib/python" PY_VER "/site-packages/certifi/cacert.pem");
    snprintf(script, sizeof(script), "%s/app.py", app_dir);

    const char *home = getenv("HOME");
    char data_dir[PATH_MAX];
    snprintf(data_dir, sizeof(data_dir), "%s/Library/Application Support/MacWhisper",
             home ? home : "/tmp");
    setenv_default("MACWHISPER_DATA_DIR", data_dir);
    setenv_default("MACWHISPER_QWEN_PYTHON", qwen_py);
    if (access(cacert, R_OK) == 0)
        setenv_default("SSL_CERT_FILE", cacert);
    /* Never pick up a user's PYTHONPATH/PYTHONHOME: the bundle is self-contained. */
    unsetenv("PYTHONPATH");
    unsetenv("PYTHONHOME");

    char log_dir[PATH_MAX], log_file[PATH_MAX];
    snprintf(log_dir, sizeof(log_dir), "%s/logs", getenv("MACWHISPER_DATA_DIR"));
    mkdirs(log_dir);
    snprintf(log_file, sizeof(log_file), "%s/macwhisper.log", log_dir);

    if (chdir(app_dir) != 0)
        die("cannot enter bundled app directory");
    if (self_test_file && self_test_file[0])
        redirect_output(self_test_file, 1);
    else if (!self_test)
        redirect_output(log_file, 0);
    warn_if_malloc_env_missing();

    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);  /* ignore PYTHON* env and user site */
    config.buffered_stdio = 0;
    config.parse_argv = 0;
    config.install_signal_handlers = 1;

    PyStatus status;
#define SET(field, val)                                                     \
    status = PyConfig_SetBytesString(&config, &config.field, (val));        \
    if (PyStatus_Exception(status)) return fail_status(&config, status);
    SET(home, py_home);
    SET(executable, py_exe);
    SET(program_name, py_exe);
    if (self_test) {
        SET(run_command, SELF_TEST_CODE);
    } else {
        SET(run_filename, script);
    }
#undef SET

    wchar_t *wapp = Py_DecodeLocale(app_dir, NULL);
    char *py_argv[] = {self_test ? "-c" : script};
    status = PyConfig_SetBytesArgv(&config, 1, py_argv);
    if (PyStatus_Exception(status))
        return fail_status(&config, status);

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        if (PyStatus_IsExit(status))
            return status.exitcode;
        Py_ExitStatusException(status);
    }
    /* Isolated mode (safe_path) never adds the script dir: put app_dir first. */
    if (wapp) {
        PyObject *sys_path = PySys_GetObject("path");
        PyObject *p = PyUnicode_FromWideChar(wapp, -1);
        if (sys_path && p)
            PyList_Insert(sys_path, 0, p);
        Py_XDECREF(p);
    }
    PyMem_RawFree(wapp);
    return Py_RunMain();
}
