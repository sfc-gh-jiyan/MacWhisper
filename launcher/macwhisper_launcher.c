/*
 * MacWhisper.app native launcher — embeds CPython and runs the repo's app.py.
 *
 * Why native (not a "#!python" script): macOS attributes privacy permissions
 * (Accessibility, Input Monitoring, Microphone) to the app's executable. A
 * script launcher hands control to the Python interpreter, so permissions
 * land on "python3.13" (or on Terminal when started from ./run.sh). Here the
 * executable stays MacWhisper itself; Python runs in-process.
 *
 * Build-time paths (set by scripts/build_app.sh):
 *   MW_REPO      repo directory containing app.py
 *   MW_VENV_PY   repo venv interpreter (venv/bin/python3) — makes Python use
 *                the venv's site-packages via its pyvenv.cfg
 *   MW_CACERT    certifi CA bundle (SSL_CERT_FILE, as run.sh sets it)
 *
 * Code changes in the repo need no rebuild: app.py is read at every launch.
 *
 *   MacWhisper              run the app (stdout/stderr -> ~/.macwhisper/logs/macwhisper.log)
 *   MacWhisper --self-test  print the embedded runtime and exit (used by tests)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef MW_REPO
#error "MW_REPO must be defined"
#endif
#ifndef MW_VENV_PY
#error "MW_VENV_PY must be defined"
#endif
#ifndef MW_CACERT
#define MW_CACERT ""
#endif

static const char *SELF_TEST_CODE =
    "import os, sys\n"
    "print('executable=' + sys.executable)\n"
    "print('prefix=' + sys.prefix)\n"
    "print('base_prefix=' + sys.base_prefix)\n"
    "print('cwd=' + os.getcwd())\n"
    "print('MallocNanoZone=' + os.environ.get('MallocNanoZone', ''))\n"
    "print('SSL_CERT_FILE=' + os.environ.get('SSL_CERT_FILE', ''))\n"
    "import rumps, mlx_whisper, objc\n"
    "print('imports=ok')\n";

static void die(const char *what) {
    fprintf(stderr, "[MacWhisper launcher] %s: %s\n", what, strerror(errno));
    exit(1);
}

/* MallocNanoZone=0 (the mlx + AppKit heap-corruption workaround from run.sh)
 * must be in the environment before the process starts, because malloc picks
 * its zones at startup. It comes from LSEnvironment in Info.plist, which
 * LaunchServices applies for a double-click, `open` and launch at login.
 *
 * Never re-exec to set it: exec gives the process a new "pid version", and
 * macOS (RunningBoard / ControlCenter) then cannot resolve the app it
 * launched. The menu bar icon is never created and permissions do not match
 * ("process has mismatched pid version"). */
static void warn_if_malloc_env_missing(void) {
    const char *v = getenv("MallocNanoZone");
    if (!v || strcmp(v, "0") != 0)
        fprintf(stderr, "[MacWhisper launcher] MallocNanoZone=0 not set "
                        "(start the app via LaunchServices, not the binary)\n");
}

static void redirect_output_to_log(void) {
    const char *home = getenv("HOME");
    if (!home)
        return;
    char dir[PATH_MAX], log[PATH_MAX];
    snprintf(dir, sizeof(dir), "%s/.macwhisper", home);
    mkdir(dir, 0755);
    snprintf(dir, sizeof(dir), "%s/.macwhisper/logs", home);
    mkdir(dir, 0755);
    snprintf(log, sizeof(log), "%s/macwhisper.log", dir);
    int fd = open(log, O_WRONLY | O_CREAT | O_APPEND, 0644);
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
    Py_ExitStatusException(status);  /* prints the error and exits */
    return 1;
}

int main(int argc, char **argv) {
    /* MW_SELF_TEST=<file> does the self-test via `open` (LaunchServices), the
     * same path as a double-click, and writes the report to <file>. */
    const char *self_test_file = getenv("MW_SELF_TEST");
    int self_test = (argc > 1 && strcmp(argv[1], "--self-test") == 0) ||
                    (self_test_file && self_test_file[0]);

    (void)argv;
    if (MW_CACERT[0] && access(MW_CACERT, R_OK) == 0)
        setenv("SSL_CERT_FILE", MW_CACERT, 1);
    if (chdir(MW_REPO) != 0)
        die("cannot enter the MacWhisper repo (" MW_REPO ")");
    if (self_test && self_test_file && self_test_file[0]) {
        int fd = open(self_test_file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            close(fd);
        }
    } else if (!self_test) {
        redirect_output_to_log();
    }
    warn_if_malloc_env_missing();

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.buffered_stdio = 0;      /* log lines appear immediately */
    config.parse_argv = 0;          /* app arguments are not Python options */

    PyStatus status = PyConfig_SetBytesString(&config, &config.executable, MW_VENV_PY);
    if (PyStatus_Exception(status))
        return fail_status(&config, status);
    status = PyConfig_SetBytesString(&config, &config.program_name, MW_VENV_PY);
    if (PyStatus_Exception(status))
        return fail_status(&config, status);

    if (self_test) {
        status = PyConfig_SetBytesString(&config, &config.run_command, SELF_TEST_CODE);
    } else {
        status = PyConfig_SetBytesString(&config, &config.run_filename, MW_REPO "/app.py");
    }
    if (PyStatus_Exception(status))
        return fail_status(&config, status);

    char *py_argv[] = {self_test ? "-c" : MW_REPO "/app.py"};
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
    return Py_RunMain();
}
