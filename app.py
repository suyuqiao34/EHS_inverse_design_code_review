from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import threading
import time
import io
import sys
import traceback
import pandas as pd
import re
import gc

# Import local modules
from multi_agent_selector import MultiAgentController

# Import compatibility bonus module (for final post-processing sorting)
try:
    from apply_compatibility_bonus_standalone import (
        load_structured_meta,
        apply_compatibility_bonus_to_dataframe
    )
    COMPATIBILITY_BONUS_AVAILABLE = True
except ImportError:
    COMPATIBILITY_BONUS_AVAILABLE = False

app = Flask(__name__, template_folder="templates")

# Disable Flask auto-reload (may cause issues on Windows)
app.config['DEBUG'] = False
app.config['PROPAGATE_EXCEPTIONS'] = True

# Directory setup
BASE_DIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = BASE_DIR  # Output files in project root: salt_combinations_formatted.csv, agent_recommended_results.csv
os.makedirs(DATA_DIR, exist_ok=True)

# Run state cache
run_state = {
    "running": False,
    "start_time": None,
    "end_time": None,
    "ok": False,
    "error": "",
    "log": "",
    "progress": 0,  # Progress percentage 0-100
    "current_step": "",  # Current step description
    "total_steps": 0,  # Total number of steps
    "completed_steps": 0,  # Number of completed steps
    "estimated_remaining": None,  # Estimated remaining time (seconds)
    "step_times": [],  # Duration record for each step
    "step_start_time": None,  # Current step start time
    "detailed_progress": []  # Detailed progress records
}

class StreamLogger(io.StringIO):
    def __init__(self, on_write):
        super().__init__()
        self.on_write = on_write
        self._buffer = []
        self._last_flush = time.time()

    def write(self, s):
        try:
            # Skip empty strings
            if s and s.strip():
                self.on_write(s)
                # Periodically flush to prevent buffer buildup
                if time.time() - self._last_flush > 0.1:
                    self.flush()
                    self._last_flush = time.time()
            return len(s)
        except Exception as e:
            # Catch log write errors to avoid interrupting main flow
            print(f"Log write failed: {e}", file=sys.__stderr__)
            return 0
    
    def flush(self):
        try:
            super().flush()
        except:
            pass

def _reset_state():
    run_state["running"] = False
    run_state["start_time"] = None
    run_state["end_time"] = None
    run_state["ok"] = False
    run_state["error"] = ""
    run_state["log"] = ""
    run_state["progress"] = 0
    run_state["current_step"] = ""
    run_state["total_steps"] = 0
    run_state["completed_steps"] = 0
    run_state["estimated_remaining"] = None
    run_state["step_times"] = []
    run_state["step_start_time"] = None
    run_state["detailed_progress"] = []

def _append_log(msg: str):
    """Append log with error protection"""
    try:
        # Limit log length to avoid excessive memory usage
        run_state["log"] += str(msg)
        if len(run_state["log"]) > 200000:
            run_state["log"] = run_state["log"][-150000:]
            # Periodically clean up memory
            gc.collect()
    except Exception as e:
        # Log write failure should not affect main flow
        print(f"Append log failed: {e}", file=sys.__stderr__)

def _update_progress(step_name: str, completed_steps: int = None, total_steps: int = None, detail: str = ""):
    """Update progress information"""
    current_time = time.time()
    
    # Record the completion time of the previous step
    if run_state["step_start_time"] and run_state["current_step"]:
        step_duration = current_time - run_state["step_start_time"]
        run_state["step_times"].append(step_duration)
        
        # Record detailed progress
        run_state["detailed_progress"].append({
            "step": run_state["current_step"],
            "duration": step_duration,
            "timestamp": current_time
        })
    
    # Update current step information
    run_state["current_step"] = step_name
    run_state["step_start_time"] = current_time

    # Preserve total steps count on repeated calls
    if total_steps is not None:
        run_state["total_steps"] = total_steps
    elif run_state["total_steps"] == 0:
        run_state["total_steps"] = 1

    # Ensure completed steps count does not decrease
    if completed_steps is not None:
        run_state["completed_steps"] = max(run_state["completed_steps"], completed_steps)

    # If completed count is not explicitly passed, assume one step forward (unless not started)
    if completed_steps is None and run_state["total_steps"] > 0 and run_state["completed_steps"] < run_state["total_steps"]:
        run_state["completed_steps"] += 1

    # Calculate progress percentage only when total steps is valid
    if run_state["total_steps"] > 0:
        run_state["completed_steps"] = min(run_state["completed_steps"], run_state["total_steps"])
        run_state["progress"] = min(100, (run_state["completed_steps"] / run_state["total_steps"]) * 100)
    
    # Estimate remaining time
    if run_state["completed_steps"] > 0 and run_state["step_times"]:
        avg_step_time = sum(run_state["step_times"]) / len(run_state["step_times"])
        remaining_steps = run_state["total_steps"] - run_state["completed_steps"]
        run_state["estimated_remaining"] = avg_step_time * remaining_steps
    
    # Output progress log
    progress_msg = f"\n{'='*60}\n"
    progress_msg += f"[Progress {run_state['progress']:.1f}%] Step {run_state['completed_steps']}/{run_state['total_steps']}: {step_name}\n"
    
    if detail:
        progress_msg += f"Detail: {detail}\n"
    
    if run_state["estimated_remaining"]:
        remaining_min = int(run_state["estimated_remaining"] // 60)
        remaining_sec = int(run_state["estimated_remaining"] % 60)
        progress_msg += f"Estimated remaining: {remaining_min}min {remaining_sec}s\n"
    
    # Show elapsed time
    if run_state["start_time"]:
        elapsed = current_time - run_state["start_time"]
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        progress_msg += f"Elapsed: {elapsed_min}min {elapsed_sec}s\n"
    
    progress_msg += f"{'='*60}\n"
    
    _append_log(progress_msg)

def _format_time_duration(seconds):
    """Format time duration"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}min {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}min"

def _preprocess_temperature_input(tm_input: str) -> str:
    """Preprocess temperature input, ensuring units are present"""
    tm_input = str(tm_input).strip()
    
    # If units already present, return as-is
    if '℃' in tm_input or '°C' in tm_input:
        return tm_input
    
    # Try to parse bare numbers and add units
    try:
        # Remove possible existing unit symbols
        clean_input = tm_input.replace('℃', '').replace('°C', '').strip()
        
        # Check for pure numeric input
        if clean_input.replace('.', '').replace('-', '').isdigit():
            return f"{clean_input}℃"
        
        # Check range format (e.g. "30-40" or "30~40")
        range_match = re.search(r'(\d+(?:\.\d+)?)\s*[~\-]\s*(\d+(?:\.\d+)?)', clean_input)
        if range_match:
            return f"{range_match.group(1)}~{range_match.group(2)}℃"
        
        # Check upper threshold format (e.g. ">30", "greater than 30", or Chinese input for "greater than")
        threshold_match = re.search(r'(?:大于|greater than|>)\s*(\d+(?:\.\d+)?)', clean_input)
        if threshold_match:
            return f">{threshold_match.group(1)}℃"
        
        # Check lower threshold format (e.g. "<30", "less than 30", or Chinese input for "less than")
        lower_threshold_match = re.search(r'(?:小于|less than|<)\s*(\d+(?:\.\d+)?)', clean_input)
        if lower_threshold_match:
            return f"<{lower_threshold_match.group(1)}℃"
        
        # If format is unrecognized, try treating as a bare number
        float(clean_input)
        return f"{clean_input}℃"
        
    except ValueError:
        # If parsing fails, return original input
        return tm_input

def _preprocess_enthalpy_input(dh_input: str) -> str:
    """Preprocess enthalpy input, ensuring units are present"""
    dh_input = str(dh_input).strip()
    
    # If units already present, return as-is
    if 'J/g' in dh_input:
        return dh_input
    
    # Try to parse and add units
    try:
        # Remove possible existing units
        clean_input = dh_input.replace('J/g', '').strip()
        
        # Check upper threshold format (e.g. ">150", "greater than 150", or Chinese input for "greater than")
        threshold_match = re.search(r'(?:大于|greater than|>)\s*(\d+(?:\.\d+)?)', clean_input)
        if threshold_match:
            return f">{threshold_match.group(1)}J/g"
        
        # Check lower threshold format (e.g. "<150", "less than 150", or Chinese input for "less than")
        lower_threshold_match = re.search(r'(?:小于|less than|<)\s*(\d+(?:\.\d+)?)', clean_input)
        if lower_threshold_match:
            return f"<{lower_threshold_match.group(1)}J/g"
        
        # Check range format
        range_match = re.search(r'(\d+(?:\.\d+)?)\s*[~\-]\s*(\d+(?:\.\d+)?)', clean_input)
        if range_match:
            return f"{range_match.group(1)}~{range_match.group(2)}J/g"
        
        # Check for pure numeric input
        if clean_input.replace('.', '').replace('-', '').isdigit():
            return f"{clean_input}J/g"
        
        # Validate as a number
        float(clean_input)
        return f"{clean_input}J/g"
        
    except ValueError:
        # If parsing fails, return original input
        return dh_input

def _run_pipeline_thread(input_path: str, tm_input: str, dh_input: str):
    """Pipeline thread function, using the multi-agent system"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        run_state["running"] = True
        run_state["start_time"] = time.time()
        run_state["ok"] = False
        run_state["error"] = ""
        run_state["log"] = ""

        # Capture console output
        sys.stdout = StreamLogger(_append_log)
        sys.stderr = StreamLogger(_append_log)
        
        # Set maximum runtime (20 minutes)
        max_runtime = 1200
        start_time = time.time()
        
        try:
            # Use multi-agent system
            _append_log("\n🚀 === Starting multi-agent salt screening system === 🚀\n")
            
            _update_progress("Initializing multi-agent system", 0, 8, "Starting multi-agent screening engine...")
            controller = MultiAgentController(input_path)
            
            _update_progress("Analysis Agent: Parsing user input", 1, 8, "Using DashScope API to parse target parameters and formulate strategy")
            _append_log(f"📝 User input: Tm={tm_input}, ΔH={dh_input}\n")
            
            _update_progress("Retrieval Agent: Screening candidate salts", 2, 8, "Rule-based filtering with LLM-optimized candidate selection")
            
            _update_progress("Compatibility Agent: Evaluating eutectic structure", 3, 8, "Using LLM to evaluate salt combination compatibility")
            
            _update_progress("Ratio Funnel Agent: Generating ratio schemes", 4, 8, "Generating multiple ratio schemes for binary or ternary combinations")
            
            _update_progress("Ternary Agent: XGBoost prediction and validation", 5, 8, "Using XGBoost model to predict physicochemical properties")
            
            # Check timeout
            if time.time() - start_time > max_runtime:
                raise TimeoutError(f"Task execution timeout (exceeded {max_runtime}s)")
            
            # Run multi-agent pipeline
            result = controller.run(tm_input, dh_input, max_retries=3)
            
            _update_progress("Decision Analysis: Evaluating result quality", 6, 8, "Analyzing predictions and classifying quality")
            
            # Force garbage collection
            gc.collect()
            
            # Save results
            if result.get('success'):
                _update_progress("Saving Results: Outputting screening results", 7, 8, "Saving screening results to CSV files")
                _save_multi_agent_results(result)
                
                # Compatibility bonus step (step 8, independent post-processing, does not affect screening results)
                _update_progress("Eutectic Compatibility Sorting: Applying compatibility bonus", 8, 8, "Re-sorting screening results by eutectic compatibility")
                _apply_compatibility_bonus_postprocessing()
                
                categorized = result.get('categorized_results', {})
                
                _append_log(f"\n🎉 Screening successful! Found {len(result.get('candidates', []))} candidate combinations\n")
                
                # Output classification statistics
                perfect_count = len(categorized.get('perfect', []))
                good_count = len(categorized.get('good', []))
                acceptable_count = len(categorized.get('acceptable', []))
                
                _append_log("\n📈 Result classification statistics:\n")
                _append_log(f"  🏆 Perfect compliance combinations: {perfect_count}\n")
                _append_log(f"  🥈 90% compliance combinations: {good_count}\n")
                _append_log(f"  🥉 80% compliance combinations: {acceptable_count}\n")
                
                if perfect_count > 0:
                    best = categorized['perfect'][0]
                    _append_log("\n🌟 Best perfect compliance combination:\n")
                    _append_log(f"   {best.get('salt_a', '')} (ratio: {best.get('ratio_a', 0):.3f})\n")
                    _append_log(f" + {best.get('salt_b', '')} (ratio: {best.get('ratio_b', 0):.3f})\n")
                    if 'salt_c' in best:
                        _append_log(f" + {best.get('salt_c', '')} (ratio: {best.get('ratio_c', 0):.3f})\n")
                    _append_log(f" → Predicted Tm: {best.get('tm_pred', 0):.1f}°C\n")
                    _append_log(f" → Predicted ΔH: {best.get('dh_pred', 0):.1f}J/g\n")
                
                _append_log(f"\n✅ Multi-agent system execution completed (Retry count: {result.get('retries', 0)})\n")
            else:
                _append_log(f"❌ Multi-agent screening failed: {result.get('message', 'Unknown reason')}\n")
                _append_log(f"Attempted {result.get('retries', 0)} retries\n")
                
                # Save partial results if available
                if len(result.get('candidates', [])) > 0:
                    _update_progress("Saving partial results", 7, 7, "Saving partial candidate combinations found")
                    _save_multi_agent_results(result)
                    _append_log(f"Saved {len(result['candidates'])} candidate combinations for reference\n")

            run_state["ok"] = result.get('success', False)
            
            # Calculate total elapsed time
            total_time = time.time() - run_state["start_time"]
            _append_log("\n🎊 === Task completed === 🎊\n")
            _append_log(f"⏱️ Total time: {_format_time_duration(total_time)}\n")
            _append_log("📁 Results saved to project root directory\n")
            
        except TimeoutError as te:
            run_state["error"] = f"Timeout error: {te}"
            _append_log("\n⏰ === Task timeout === ⏰\n")
            _append_log(run_state["error"])
            _append_log("\nPlease try reducing data size or relaxing screening criteria\n")
        except MemoryError as me:
            run_state["error"] = f"Out of memory: {me}"
            _append_log("\n💾 === Out of memory === 💾\n")
            _append_log(run_state["error"])
            _append_log("\nPlease try reducing data or closing other programs\n")
        except Exception as e:
            run_state["error"] = f"{e}\n" + traceback.format_exc()
            _append_log("\n💥 === Task failed === 💥\n")
            _append_log(run_state["error"])
        finally:
            # Restore standard output
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except:
                pass
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as e:
        # Outer exception handler (prevent thread crash)
        try:
            run_state["error"] = f"Thread exception: {e}\n" + traceback.format_exc()
            _append_log("\n💥 === Thread exception === 💥\n")
            _append_log(run_state["error"])
        except:
            # If even logging fails, at least output to console
            print(f"Critical error: {e}", file=sys.__stderr__)
            traceback.print_exc(file=sys.__stderr__)
    finally:
        # Ensure state is properly set
        try:
            run_state["running"] = False
            run_state["end_time"] = time.time()
            # Final cleanup
            gc.collect()
        except:
            pass
        
        # Ensure output is restored
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def _save_multi_agent_results(result: dict):
    """Save multi-agent system results to CSV files"""
    try:
        candidates = result.get('candidates', [])
        categorized_results = result.get('categorized_results', {})
        
        if not candidates:
            _append_log("⚠️ Warning: No results to save\n")
            return

        # Get all valid candidates (>=80% compliance)
        filtered_candidates = (
            categorized_results.get('perfect', []) + 
            categorized_results.get('good', []) + 
            categorized_results.get('acceptable', [])
        )
        
        # If no high-quality results, save all candidates for reference
        if not filtered_candidates:
            _append_log("⚠️ Warning: No results with >=80% compliance, saving all candidates for reference\n")
            filtered_candidates = candidates
        
        # Convert to DataFrame
        df = pd.DataFrame(filtered_candidates)

        # Save recommended results
        rec_path = os.path.join(OUTPUT_DIR, "agent_recommended_results.csv")
        df.to_csv(rec_path, index=False, encoding='utf-8-sig')
        _append_log(f"💾 Saved recommended results: {len(filtered_candidates)} combinations → {rec_path}\n")
        
        # Save categorized results (only non-empty levels)
        for level, level_candidates in categorized_results.items():
            if level in ['perfect', 'good', 'acceptable'] and level_candidates:
                level_df = pd.DataFrame(level_candidates)
                level_path = os.path.join(OUTPUT_DIR, f"results_{level}_compliance.csv")
                level_df.to_csv(level_path, index=False, encoding='utf-8-sig')
                _append_log(f"💾 Saved {level} compliance results: {len(level_candidates)} combinations\n")

        # Create formatted file (compatible with existing download functionality)
        formatted_path = os.path.join(OUTPUT_DIR, "salt_combinations_formatted.csv")
        
        # Ensure required columns exist
        basic_cols = ['salt_a', 'salt_b']
        if 'salt_c' in df.columns:
            basic_cols.append('salt_c')
        basic_cols.extend(['ratio_a', 'ratio_b'])
        if 'ratio_c' in df.columns:
            basic_cols.append('ratio_c')
        basic_cols.extend(['tm_pred', 'dh_pred', 'single_peak_proba', 'score'])
        
        # Add optional columns
        optional_cols = ['compliance_level', 'compliance_desc', 'type', 'is_single_peak']
        for col in optional_cols:
            if col in df.columns:
                basic_cols.append(col)
        
        # Select only existing columns
        available_cols = [col for col in basic_cols if col in df.columns]
        simple_df = df[available_cols].copy()
        simple_df.to_csv(formatted_path, index=False, encoding='utf-8-sig')
        _append_log(f"💾 Saved formatted results: {len(simple_df)} combinations → {formatted_path}\n")

    except Exception as e:
        _append_log(f"❌ Error saving multi-agent results: {e}\n")
        import traceback
        _append_log(traceback.format_exc())

def _apply_compatibility_bonus_postprocessing():
    """Apply eutectic compatibility bonus sorting (independent post-processing step, does not affect screening results)"""
    try:
        if not COMPATIBILITY_BONUS_AVAILABLE:
            _append_log("⚠️ Warning: Compatibility bonus module unavailable, skipping compatibility sorting\n")
            _append_log("   Note: Please ensure apply_compatibility_bonus_standalone.py exists\n")
            return
        
        _append_log(f"\n{'='*60}\n")
        _append_log("🔬 Applying eutectic compatibility bonus sorting (post-processing step)\n")
        _append_log(f"{'='*60}\n")
        
        # Load metadata
        _append_log("📚 Loading salt metadata...\n")
        meta_map = load_structured_meta('data')
        if len(meta_map) == 0:
            _append_log("⚠️ Warning: No metadata loaded, skipping compatibility sorting\n")
            return
        
        _append_log(f"✓ Successfully loaded metadata for {len(meta_map)} salts\n")
        
        # Process main results file
        rec_path = os.path.join(OUTPUT_DIR, "agent_recommended_results.csv")
        if not os.path.exists(rec_path):
            _append_log("⚠️ Warning: Main results file not found, skipping compatibility sorting\n")
            return
        
        # Read results file
        df_with_bonus = pd.read_csv(rec_path, encoding='utf-8-sig')
        
        # Apply compatibility bonus (recalculates score and re-sorts, but does not change screening results)
        _append_log("🔄 Calculating compatibility bonus and re-sorting...\n")
        df_with_bonus = apply_compatibility_bonus_to_dataframe(df_with_bonus, meta_map)
        
        # Save updated results (overwrite original file)
        df_with_bonus.to_csv(rec_path, index=False, encoding='utf-8-sig')
        _append_log(f"✓ Updated main results file: {rec_path}\n")
        
        # Update categorized results files (also apply compatibility bonus)
        _append_log("🔄 Updating categorized results with compatibility sorting...\n")
        for level in ['perfect', 'good', 'acceptable']:
            level_path = os.path.join(OUTPUT_DIR, f"results_{level}_compliance.csv")
            if os.path.exists(level_path):
                level_df = pd.read_csv(level_path, encoding='utf-8-sig')
                level_df = apply_compatibility_bonus_to_dataframe(level_df, meta_map)
                level_df.to_csv(level_path, index=False, encoding='utf-8-sig')
                _append_log(f"✓ Updated {level} compliance results file\n")
        
        # Update formatted file (use data with compatibility bonus applied, but keep original columns only)
        formatted_path = os.path.join(OUTPUT_DIR, "salt_combinations_formatted.csv")
        if os.path.exists(formatted_path):
            # Determine columns to keep (read column names from original file)
            original_formatted_df = pd.read_csv(formatted_path, encoding='utf-8-sig', nrows=0)
            formatted_cols_to_keep = [col for col in original_formatted_df.columns if col in df_with_bonus.columns]
            formatted_df = df_with_bonus[formatted_cols_to_keep].copy()
            formatted_df.to_csv(formatted_path, index=False, encoding='utf-8-sig')
            _append_log(f"✓ Updated formatted results file: {formatted_path}\n")
        
        _append_log(f"\n{'='*60}\n")
        _append_log("✓ Eutectic compatibility bonus sorting completed!\n")
        _append_log("  - Updated all result files\n")
        _append_log("  - Results re-sorted by eutectic compatibility\n")
        _append_log(f"{'='*60}\n\n")
        
    except Exception as e:
        # Compatibility bonus failure should not affect main flow
        _append_log(f"⚠️ Warning: Error applying compatibility bonus: {e}\n")
        _append_log("   But screening results have been saved, main functionality unaffected\n")
        import traceback
        _append_log(traceback.format_exc() + "\n")

    except Exception as e:
        _append_log(f"❌ Error saving multi-agent results: {e}\n")
        import traceback
        _append_log(traceback.format_exc())

@app.route("/", methods=["GET"])
def index():
    # Check if output files exist
    formatted_path = os.path.join(OUTPUT_DIR, "salt_combinations_formatted.csv")
    rec_path = os.path.join(OUTPUT_DIR, "agent_recommended_results.csv")
    perfect_path = os.path.join(OUTPUT_DIR, "results_perfect_compliance.csv")
    good_path = os.path.join(OUTPUT_DIR, "results_good_compliance.csv")
    acceptable_path = os.path.join(OUTPUT_DIR, "results_acceptable_compliance.csv")

    return render_template(
        "index.html",
        running=run_state["running"],
        ok=run_state["ok"],
        error=run_state["error"],
        formatted_exists=os.path.exists(formatted_path),
        rec_exists=os.path.exists(rec_path),
        perfect_exists=os.path.exists(perfect_path),
        good_exists=os.path.exists(good_path),
        acceptable_exists=os.path.exists(acceptable_path)
    )

@app.route("/upload_run", methods=["POST"])
def upload_run():
    try:
        if run_state["running"]:
            return jsonify({"status": "running", "message": "Task is currently running, please try again later"}), 409

        file = request.files.get("file")
        tm = request.form.get("tm")
        dh = request.form.get("dh")
        if not file or file.filename == "":
            return jsonify({"status": "error", "message": "Please upload a CSV file"}), 400
        if not tm or not dh:
            return jsonify({"status": "error", "message": "Please enter target Tm and ΔH"}), 400

        # Save to data directory
        save_name = "uploaded_input.csv"
        input_path = os.path.join(DATA_DIR, save_name)
        file.save(input_path)

        # Pre-check: can we read it?
        try:
            _ = pd.read_csv(input_path, nrows=5, encoding="utf-8")
        except Exception:
            # Fallback to gbk encoding
            try:
                _ = pd.read_csv(input_path, nrows=5, encoding="gbk")
            except Exception:
                return jsonify({"status": "error", "message": "Cannot read CSV, please check encoding or format"}), 400

        _reset_state()
        # Start pipeline thread (do not set daemon=True to ensure completion)
        t = threading.Thread(
            target=_run_pipeline_thread, 
            args=(input_path, tm, dh),
            name="PipelineThread"
        )
        t.daemon = False  # Do not run as a daemon thread; allow normal completion
        t.start()
        
        # Wait briefly to ensure the thread starts successfully
        time.sleep(0.1)
        if not t.is_alive():
            return jsonify({"status": "error", "message": "Thread startup failed"}), 500
            
        return jsonify({"status": "ok", "message": "Task started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    result = {
        "running": run_state["running"],
        "ok": run_state["ok"],
        "error": run_state["error"],
        "log_tail": run_state["log"][-4000:] if run_state["log"] else "",
        "progress": run_state["progress"],
        "current_step": run_state["current_step"],
        "completed_steps": run_state["completed_steps"],
        "total_steps": run_state["total_steps"],
        "estimated_remaining": run_state["estimated_remaining"]
    }
    
    # Add formatted remaining time
    if run_state["estimated_remaining"]:
        result["estimated_remaining_formatted"] = _format_time_duration(run_state["estimated_remaining"])
    
    # Add elapsed time
    if run_state["start_time"]:
        elapsed = time.time() - run_state["start_time"]
        result["elapsed_time"] = elapsed
        result["elapsed_time_formatted"] = _format_time_duration(elapsed)
    
    # If the task completed successfully, add result statistics
    if run_state["ok"] and not run_state["running"]:
        try:
            rec_path = os.path.join(OUTPUT_DIR, "agent_recommended_results.csv")
            if os.path.exists(rec_path):
                df = pd.read_csv(rec_path)
                result["stats"] = {
                    "total_combinations": len(df),
                    "tm_range": [float(df['tm_pred'].min()), float(df['tm_pred'].max())],
                    "dh_range": [float(df['dh_pred'].min()), float(df['dh_pred'].max())],
                    "peak_prob_range": [float(df['single_peak_proba'].min()), float(df['single_peak_proba'].max())],
                    "binary_count": len(df[df['type'] == 'binary']) if 'type' in df.columns else 0,
                    "ternary_count": len(df[df['type'] == 'ternary']) if 'type' in df.columns else 0
                }
        except Exception as e:
            print(f"Failed to get statistics: {e}")
    
    return jsonify(result)

@app.route("/download/<string:which>", methods=["GET", "HEAD"])
def download(which: str):
    file_mapping = {
        "features": "salt_combinations_formatted.csv",
        "recommend": "agent_recommended_results.csv",
        "perfect": "results_perfect_compliance.csv",
        "good": "results_good_compliance.csv", 
        "acceptable": "results_acceptable_compliance.csv"
    }
    
    if which not in file_mapping:
        return "invalid file", 404
    
    fn = file_mapping[which]
    path = os.path.join(OUTPUT_DIR, fn)
    
    if not os.path.exists(path):
        return f"File not found: {fn}, please run a task first.", 404
    
    # For HEAD requests, just return 200 if file exists
    if request.method == "HEAD":
        return "", 200
    
    return send_from_directory(OUTPUT_DIR, fn, as_attachment=True)

if __name__ == "__main__":
    # Disable debug mode on Windows to avoid process issues
    # Use threaded=True to enable multi-threading
    print("=" * 60)
    print("🚀 Starting Salt Screening Web Service")
    print("=" * 60)
    print("📍 Access URL: http://localhost:5000")
    print("📍 Or use: http://127.0.0.1:5000")
    print("💡 Note: For LAN access, use http://0.0.0.0:5000")
    print("=" * 60)
    
    try:
        app.run(
            host="0.0.0.0", 
            port=5000, 
            debug=False,  # Disable debug mode on Windows
            threaded=True,  # Enable multi-threading support
            use_reloader=False  # Disable auto-reload
        )
    except KeyboardInterrupt:
        print("\n👋 Service stopped")
    except Exception as e:
        print(f"\n❌ Service startup failed: {e}")
        traceback.print_exc()

