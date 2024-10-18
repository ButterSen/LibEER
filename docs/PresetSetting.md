<div>
<table border="0" cellpadding="0" cellspacing="0" width="818" style="border-collapse:
 collapse;table-layout:fixed;width:612pt">
 <colgroup><col width="109" style="mso-width-source:userset;mso-width-alt:3474;width:81pt">
 <col width="122" style="mso-width-source:userset;mso-width-alt:3894;width:91pt">
 <col width="587" style="mso-width-source:userset;mso-width-alt:18779;width:440pt">
 </colgroup><tbody><tr height="20" style="height:15.0pt">
  <td rowspan="18" height="989" class="xl70" width="109" style="height:740.7pt;
  width:81pt">SEED Dataset</td>
  <td colspan="2" class="xl67" width="709" style="border-right:.5pt solid black;
  border-left:none;width:531pt;white-space:no-wrap">subject-dependent, train
  val test split<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl69" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_dependent_train_val_test_setting</td>
 </tr>
 <tr height="132" style="height:99.0pt">
  <td height="132" class="xl65" style="height:99.0pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  the 15 subjects in the dataset, perform one round of training and testing for
  each individual subject’s session data. Specifically, randomly select any 9
  trials from the 15 trials in a session for the training set, 3 trials for the
  validation set, and 3 trials for the test set.<br>
    If using data from two sessions, then for one subject’s session, randomly
  select any 9 trials from the 15 trials as the training set, 3 trials as the
  validation set, and 3 trials as the test set for one training session. In
  total, this results in 30 training sessions (num_session * num_subject: 2 *
  15).</td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">subject-independent,<span style="mso-spacerun:yes">&nbsp; </span>train val test split<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl69" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_independent_train_val_test_setting</td>
 </tr>
 <tr height="38" style="height:28.3pt">
  <td height="38" class="xl65" style="height:28.3pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt"><span style="font-variant-ligatures: normal;font-variant-caps: normal;orphans: 2;
  white-space:pre-wrap;widows: 2;-webkit-text-stroke-width: 0px;text-decoration-thickness: initial;
  text-decoration-style: initial;text-decoration-color: initial">For the 15
  subjects in the dataset, randomly select 9 subjects from the first session as
  the training set, 3 subjects as the validation set, and 3 subjects as the
  test set.</span></td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">subject-dependent, front
  nine trials and back six trials<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl65" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_dependent_front_back_setting</td>
 </tr>
 <tr height="151" style="height:113.15pt">
  <td height="151" class="xl65" style="height:113.15pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  the 15 subjects in the dataset, a round of training and testing is performed
  for each subject's session data. Specifically, for one session of a subject,
  the first 9 trials out of the 15 trials are used as the training set, and the
  last 6 trials as the test set for training.<br>
    If using data from two sessions, the first 9 trials out of 15 from one
  session of a subject are used as the training set, and the last 6 trials as
  the test set for one round of training, resulting in a total of 30 training
  rounds (num_session * num_subject: 2 * 15). Therefore, using data from three
  sessions requires 45 training rounds, while one session requires 15 training
  rounds.</td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">subject-dependent, five
  fold cross-validation<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl65" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_dependent_5fold_setting</td>
 </tr>
 <tr height="189" style="height:141.45pt">
  <td height="189" class="xl65" style="height:141.45pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  the 15 subjects in the dataset, perform five rounds of training and testing
  for each individual subject's session data. Specifically, for one session of
  a subject, take three trials sequentially from the 15 trials as the training
  set, while the remaining trials serve as the test set for one training
  session. This process continues until all trials have been used as the test
  set, resulting in a total of five training sessions.<br>
    If using data from two sessions, for one subject’s session with 15 trials,
  first take the first three trials as the test set and the remaining 12 trials
  as the training set for one session. Then take trials 4, 5, and 6 as the test
  set, with the other 12 trials as the training set for another session. This
  continues until all trials have been used as the test set. The data from two
  sessions will require a total of 2 * 15 * 5 training rounds.</td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">subject-independent,
  leave one out:</td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl65" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_sub_independent_leave_one_out_setting</td>
 </tr>
 <tr height="170" style="height:127.3pt">
  <td height="170" class="xl65" style="height:127.3pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  all the data in the dataset, perform 15 training rounds using all 15 subjects
  from one session. Specifically, use the data from one subject in a session as
  the test set, while the data from the remaining subjects serves as the
  training set for one training round. This process continues until all
  subjects have been used as the test set, resulting in a total of fifteen
  training rounds.<br>
    If using data from two sessions, start with the data from the first
  session, alternately selecting one subject's data as the test set while using
  the data from the other subjects as the training set for training. This
  results in 15 training sessions for one session. With two sessions, a total
  of 2 * 15 training rounds will be conducted.</td>
 </tr>
 <tr height="20" style="height:15.0pt">
  <td colspan="2" height="20" class="xl67" style="border-right:.5pt solid black;
  height:15.0pt;border-left:none;white-space:no-wrap">cross-session<font class="font8">：</font></td>
 </tr>
 <tr height="19" style="height:14.15pt">
  <td height="19" class="xl65" style="height:14.15pt;border-top:none;border-left:
  none;white-space:no-wrap">params</td>
  <td class="xl65" style="border-top:none;border-left:none;white-space:no-wrap">-setting
  seed_cross_session_setting</td>
 </tr>
 <tr height="75" style="height:56.6pt">
  <td height="75" class="xl65" style="height:56.6pt;border-top:none;border-left:
  none;white-space:no-wrap">description</td>
  <td class="xl66" width="587" style="border-top:none;border-left:none;width:440pt">For
  all the data in the dataset, alternately select the data from one session as
  the test set while using the data from the other two sessions as the training
  set, continuing this process until all sessions have been used as the test
  set. A total of three training sessions will be conducted.</td>
 </tr>
 <!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="109" style="width:81pt"></td>
  <td width="122" style="width:91pt"></td>
  <td width="587" style="width:440pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>
</div>

