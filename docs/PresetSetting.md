<div>
<table border="0" cellpadding="0" cellspacing="0" width="818" style="border-collapse:
 collapse;table-layout:fixed;width:614pt">
 <colgroup><col width="109" style="mso-width-source:userset;mso-width-alt:3488;width:82pt">
 <col width="122" style="mso-width-source:userset;mso-width-alt:3904;width:92pt">
 <col width="587" style="mso-width-source:userset;mso-width-alt:18784;width:440pt">
 </colgroup><tbody><tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td rowspan="18" height="1210" class="xl76" width="109" style="border-bottom:.5pt solid black;
  height:908.7pt;width:82pt">SEED Dataset</td>
  <td colspan="2" class="xl79" width="709" style="border-right:.5pt solid black;
  border-left:none;width:532pt;white-space:no-wrap">subject-dependent, train
  val test split<font class="font8">：</font></td>
 </tr>
 <tr height="22" style="height:16.5pt">
  <td height="22" class="xl65" style="height:16.5pt;white-space:no-wrap">params</td>
  <td class="xl66" style="white-space:no-wrap">-setting
  seed_sub_dependent_train_val_test_setting</td>
 </tr>
 <tr height="140" style="height:105.0pt">
  <td height="140" class="xl67" style="height:105.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt">For the 15 subjects in the
  dataset, perform one round of training and testing for each individual
  subject’s session data. Specifically, randomly select any 9 trials from the
  15 trials in a session for the training set, 3 trials for the validation set,
  and 3 trials for the test set.<br>
    If using data from two sessions, then for one subject’s session, randomly
  select any 9 trials from the 15 trials as the training set, 3 trials as the
  validation set, and 3 trials as the test set for one training session. In
  total, this results in 30 training sessions (num_session * num_subject: 2 *
  15).</td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td colspan="2" height="53" class="xl79" style="border-right:.5pt solid black;
  height:39.95pt;border-left:none;white-space:no-wrap">subject-independent,<span style="mso-spacerun:yes">&nbsp; </span>train val test split<font class="font8">：</font></td>
 </tr>
 <tr height="22" style="height:16.5pt">
  <td height="22" class="xl65" style="height:16.5pt;white-space:no-wrap">params</td>
  <td class="xl66" style="white-space:no-wrap">-setting
  seed_sub_independent_train_val_test_setting</td>
 </tr>
 <tr height="40" style="height:30.0pt">
  <td height="40" class="xl67" style="height:30.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt"><span style="font-variant-ligatures: normal;
  font-variant-caps: normal;orphans: 2;white-space:pre-wrap;widows: 2;
  -webkit-text-stroke-width: 0px;text-decoration-thickness: initial;text-decoration-style: initial;
  text-decoration-color: initial">For the 15 subjects in the dataset, randomly
  select 9 subjects from the first session as the training set, 3 subjects as
  the validation set, and 3 subjects as the test set.</span></td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td colspan="2" height="53" class="xl79" style="border-right:.5pt solid black;
  height:39.95pt;border-left:none;white-space:no-wrap">subject-dependent, front
  nine trials and back six trials<font class="font8">：</font></td>
 </tr>
 <tr height="22" style="height:16.5pt">
  <td height="22" class="xl65" style="height:16.5pt;white-space:no-wrap">params</td>
  <td class="xl69" style="white-space:no-wrap">-setting
  seed_sub_dependent_front_back_setting</td>
 </tr>
 <tr height="140" style="height:105.0pt">
  <td height="140" class="xl67" style="height:105.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt">For the 15 subjects in the
  dataset, a round of training and testing is performed for each subject's
  session data. Specifically, for one session of a subject, the first 9 trials
  out of the 15 trials are used as the training set, and the last 6 trials as the
  test set for training.<br>
    If using data from two sessions, the first 9 trials out of 15 from one
  session of a subject are used as the training set, and the last 6 trials as
  the test set for one round of training, resulting in a total of 30 training
  rounds (num_session * num_subject: 2 * 15). Therefore, using data from three
  sessions requires 45 training rounds, while one session requires 15 training
  rounds.</td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td colspan="2" height="53" class="xl79" style="border-right:.5pt solid black;
  height:39.95pt;border-left:none;white-space:no-wrap">subject-dependent, five
  fold cross-validation<font class="font8">：</font></td>
 </tr>
 <tr height="22" style="height:16.5pt">
  <td height="22" class="xl65" style="height:16.5pt;white-space:no-wrap">params</td>
  <td class="xl69" style="white-space:no-wrap">-setting
  seed_sub_dependent_5fold_setting</td>
 </tr>
 <tr height="200" style="height:150.0pt">
  <td height="200" class="xl67" style="height:150.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt">For the 15 subjects in the
  dataset, perform five rounds of training and testing for each individual
  subject's session data. Specifically, for one session of a subject, take
  three trials sequentially from the 15 trials as the training set, while the
  remaining trials serve as the test set for one training session. This process
  continues until all trials have been used as the test set, resulting in a
  total of five training sessions.<br>
    If using data from two sessions, for one subject’s session with 15 trials,
  first take the first three trials as the test set and the remaining 12 trials
  as the training set for one session. Then take trials 4, 5, and 6 as the test
  set, with the other 12 trials as the training set for another session. This
  continues until all trials have been used as the test set. The data from two
  sessions will require a total of 2 * 15 * 5 training rounds.</td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td colspan="2" height="53" class="xl79" style="border-right:.5pt solid black;
  height:39.95pt;border-left:none;white-space:no-wrap">subject-independent,
  leave one out:</td>
 </tr>
 <tr height="22" style="height:16.5pt">
  <td height="22" class="xl65" style="height:16.5pt;white-space:no-wrap">params</td>
  <td class="xl69" style="white-space:no-wrap">-setting
  seed_sub_independent_leave_one_out_setting</td>
 </tr>
 <tr height="180" style="height:135.0pt">
  <td height="180" class="xl67" style="height:135.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt">For all the data in the dataset,
  perform 15 training rounds using all 15 subjects from one session.
  Specifically, use the data from one subject in a session as the test set,
  while the data from the remaining subjects serves as the training set for one
  training round. This process continues until all subjects have been used as
  the test set, resulting in a total of fifteen training rounds.<br>
    If using data from two sessions, start with the data from the first
  session, alternately selecting one subject's data as the test set while using
  the data from the other subjects as the training set for training. This
  results in 15 training sessions for one session. With two sessions, a total
  of 2 * 15 training rounds will be conducted.</td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td colspan="2" height="53" class="xl79" style="border-right:.5pt solid black;
  height:39.95pt;border-left:none;white-space:no-wrap">cross-session<font class="font8">：</font></td>
 </tr>
 <tr height="22" style="height:16.5pt">
  <td height="22" class="xl65" style="height:16.5pt;white-space:no-wrap">params</td>
  <td class="xl69" style="white-space:no-wrap">-setting
  seed_cross_session_setting</td>
 </tr>
 <tr height="60" style="height:45.0pt">
  <td height="60" class="xl70" style="height:45.0pt;white-space:no-wrap">description</td>
  <td class="xl71" width="587" style="width:440pt">For all the data in the dataset,
  alternately select the data from one session as the test set while using the
  data from the other two sessions as the training set, continuing this process
  until all sessions have been used as the test set. A total of three training
  sessions will be conducted.</td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td rowspan="12" height="638" class="xl82" style="border-bottom:.5pt solid black;
  height:479.3pt;border-top:none">DEAP Dataset</td>
  <td colspan="2" class="xl79" style="border-right:.5pt solid black;border-left:
  none">subject-dependent, train-val-test:</td>
 </tr>
 <tr height="24" style="height:18.0pt">
  <td height="24" class="xl72" style="height:18.0pt">params</td>
  <td class="xl73">-setting deap_sub_dependent_train_val_test_setting</td>
 </tr>
 <tr height="110" style="height:82.5pt">
  <td height="110" class="xl67" style="height:82.5pt;white-space:no-wrap">description</td>
  <td class="xl74" width="587" style="width:440pt">For the 32 subjects in the
  dataset, perform one round of training and testing for each individual
  subject’s data. Specifically, randomly select any 24 trials from the 40
  trials for the training set, 8 trials for the validation set, and 8 trials
  for the test set.<br>
    </td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td colspan="2" height="53" class="xl79" style="border-right:.5pt solid black;
  height:39.95pt;border-left:none">subject-dependent, ten fold split:</td>
 </tr>
 <tr height="24" style="height:18.0pt">
  <td height="24" class="xl72" style="height:18.0pt">params</td>
  <td class="xl75"><span style="mso-spacerun:yes">&nbsp;</span>-setting
  deap_sub_dependent_10fold_setting</td>
 </tr>
 <tr height="100" style="height:75.0pt">
  <td height="100" class="xl67" style="height:75.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt">For the 32 subjects in the
  dataset, perform ten rounds of training and testing for each individual
  subject's data. Specifically, take 36 trials randomly from the 40 trials as
  the training set, while the remaining trials serve as the test set. This
  process continues until all trials have been used as the test set, resulting
  in a total of ten training sessions.<br>
    </td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td colspan="2" height="53" class="xl79" style="border-right:.5pt solid black;
  height:39.95pt;border-left:none">subject-independent, train-val-test:</td>
 </tr>
 <tr height="24" style="height:18.0pt">
  <td height="24" class="xl72" style="height:18.0pt">params</td>
  <td class="xl75"><span style="mso-spacerun:yes">&nbsp;</span>-setting
  deap_sub_independent_train_val_test_setting</td>
 </tr>
 <tr height="40" style="height:30.0pt">
  <td height="40" class="xl67" style="height:30.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt"><span style="font-variant-ligatures: normal;
  font-variant-caps: normal;orphans: 2;white-space:pre-wrap;widows: 2;
  -webkit-text-stroke-width: 0px;text-decoration-thickness: initial;text-decoration-style: initial;
  text-decoration-color: initial">For the 32 subjects in the dataset, randomly
  select 20 subjects as the training set, 6 subjects as the validation set, and
  6 subjects as the test set.</span></td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td colspan="2" height="53" class="xl79" style="border-right:.5pt solid black;
  height:39.95pt;border-left:none">subject-independent, leave one out:</td>
 </tr>
 <tr height="24" style="height:18.0pt">
  <td height="24" class="xl72" style="height:18.0pt">params</td>
  <td class="xl75"><span style="mso-spacerun:yes">&nbsp;</span>-setting
  deap_sub_independent_leave_one_out_setting</td>
 </tr>
 <tr height="80" style="height:60.0pt">
  <td height="80" class="xl67" style="height:60.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt">For all the data in the dataset,
  perform 40 training rounds using all 40 subjects from one session.
  Specifically, use the data from one subject as the test set, while the data
  from the remaining subjects serves as the training set for one training
  round. This process continues until all subjects have been used as the test
  set, resulting in a total of 40 training rounds.</td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td rowspan="6" height="282" class="xl82" style="border-bottom:.5pt solid black;
  height:211.9pt;border-top:none">HCI Dataset</td>
  <td colspan="2" class="xl79" style="border-right:.5pt solid black;border-left:
  none">subject-dependent, train-val-test:</td>
 </tr>
 <tr height="24" style="height:18.0pt">
  <td height="24" class="xl72" style="height:18.0pt">params</td>
  <td class="xl73">-setting hci_sub_dependent_train_val_test_setting</td>
 </tr>
 <tr height="88" style="height:66.0pt">
  <td height="88" class="xl67" style="height:66.0pt;white-space:no-wrap">description</td>
  <td class="xl74" width="587" style="width:440pt">For each of the 28 subjects in
  the dataset, conduct one round of training and testing using their individual
  data. Specifically, randomly select 60% of the trials for the training set,
  20% for the validation set, and the remaining 20% for the test set.</td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td colspan="2" height="53" class="xl79" style="border-right:.5pt solid black;
  height:39.95pt;border-left:none">subject-independent, train-val-test:</td>
 </tr>
 <tr height="24" style="height:18.0pt">
  <td height="24" class="xl72" style="height:18.0pt">params</td>
  <td class="xl75"><span style="mso-spacerun:yes">&nbsp;</span>-setting
  hci_sub_independent_train_val_test_setting</td>
 </tr>
 <tr height="40" style="height:30.0pt">
  <td height="40" class="xl67" style="height:30.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt"><span style="font-variant-ligatures: normal;
  font-variant-caps: normal;orphans: 2;white-space:pre-wrap;widows: 2;
  -webkit-text-stroke-width: 0px;text-decoration-thickness: initial;text-decoration-style: initial;
  text-decoration-color: initial">For the 28 subjects in the dataset, randomly
  select 18 subjects as the training set, 5 subjects as the validation set, and
  5 subjects as the test set.</span></td>
 </tr>
 <tr height="53" style="mso-height-source:userset;height:39.95pt">
  <td rowspan="3" height="117" class="xl82" style="border-bottom:.5pt solid black;
  height:87.95pt;border-top:none">Faced Dataset</td>
  <td colspan="2" class="xl79" style="border-right:.5pt solid black;border-left:
  none">subject-independent, train-val-test:</td>
 </tr>
 <tr height="24" style="height:18.0pt">
  <td height="24" class="xl72" style="height:18.0pt">params</td>
  <td class="xl75"><span style="mso-spacerun:yes">&nbsp;</span>-setting
  faced_sub_independent_train_val_test_setting</td>
 </tr>
 <tr height="40" style="height:30.0pt">
  <td height="40" class="xl67" style="height:30.0pt;white-space:no-wrap">description</td>
  <td class="xl68" width="587" style="width:440pt"><span style="font-variant-ligatures: normal;
  font-variant-caps: normal;orphans: 2;white-space:pre-wrap;widows: 2;
  -webkit-text-stroke-width: 0px;text-decoration-thickness: initial;text-decoration-style: initial;
  text-decoration-color: initial">For the 123 subjects in the dataset, randomly
  select 75 subjects as the training set, 24 subjects as the validation set,
  and 24 subjects as the test set.</span></td>
 </tr>
 <!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="109" style="width:82pt"></td>
  <td width="122" style="width:92pt"></td>
  <td width="587" style="width:440pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>
</div>
