CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��O�;dZ        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��_   max       Pð�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �� �   max       <��
        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @Ftz�G�     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v}p��
>     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��@            8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       <e`B        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B2&�        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B2        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >
�   max       C���        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��   max       C��G        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ]        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��_   max       P��G        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���p:�   max       ?�ۋ�q�        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       <��
        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @Ftz�G�     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @v}p��
>     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q�           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��`            \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F   max         F        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?wX�e+�   max       ?���>BZ�     �  ^�            R            $   0               :                      9   -   &   \   H      Y               B      	   >               +                     	      #   #                        &      
                  %         -         N'ӹM��_N6�{Pð�N@E�N��N���O��P h�O���N6��N��Pl�O؟�P�uO�aEN1\�N��N��N�PJ�O�RTO�p�PJ'fP|N�M3Oق�N�?O�~�NL�N9yP���N�w�N���P�O�DN�YON�AO23O��[O/��N���N��OK�O�3AO��rN���NM��O��OΉO�&N�zAN(fOU�xN��mN<�XOP�PT�O1�N��`N<�
O�'}Ñ2M��&O&!�O~lO��vN�UQO͔VO	A�N*��N4�<��
<T��<T��;�`B;��
;�o;D����o��o�ě���`B��`B��`B�t��t��#�
�D���D���u��o��o��C���C���t����㼴9X��9X��9X��j��j��j�ě�����������/��/��h��h�����������+�C��\)�\)�\)�\)�t��t��t����''0 Ž0 Ž8Q�D���T���m�h�u�y�#��o�����+��7L��hs���P������{�� ���������������������� ����������#/3<E</#�0I{������{Z<0�����������������������?BIN[[`b[NB:????????26ABDOV[g[OB96222222����
#)-.)
����������
/HQWXH
������[amz�����������zmaZ[4<HNSHE<634444444444��������������������)32BK_[^hkwth[B6 )�������
���������%+4BO[ksqhO>6)"/;HQTXTPPKE;/"#*+# #0<=BA<90(#        �����������vwz������������������������������ ������������������������������������������������05BNt�������tgNB5++0^hnz������������zka^`akmsz��{zmia```````���	
#043,!
��������������������������������)5A9)����4<HKOKH<494444444444OUaknopnjaUROOOOOOOOw���������������ztow��������������������:<IUbknrnkdbZUI<:8::p{�����������������p~������������������~imxz��������zomfiiii��������������������$/<HKTUWUTPH<<4/,%$$_nz�����������znhda_������������������������������������������������������������<BDBHOP[htrnomh[SOB<��������������������������������������������������������16<BNOYONB6211111111dt�������������ti\_d;BNg���������tg[F=9;�����������������JOT\huw���wuh\OJJJJJ$)67866-)(&#$$$$$$$$�
#/45660#
���.07<?BGILMMJI<40.++."#/'&#������
�������4HM[gt��������tgNB64��������������������^bdnsvwyxvnb`YYZ[Z\^���������������������(0<EMMIC;0$
������������������������	����������������������������������������t�������������}~wvgt���������������������)7DIH@9)
������adnozz���������zniaa )1.)'	��������������	����������������������²¬±²¿��������¿²²²²²²²²²²�H�C�G�H�P�U�_�^�`�U�H�H�H�H�H�H�H�H�H�H�����k�M�>�,�-�Z����������������������
���������
��� ���
�
�
�
�
�
�
�
�f�a�Z�U�Z�Z�f�s�~�}�u�s�f�f�f�f�f�f�f�f�a�^�U�T�U�W�a�g�n�x�z�w�n�g�a�a�a�a�a�aìá×ÖÖÕÚàìù����������������ùì�4�(����"�.�4�A�M�i�������f�Z�M�A�4�	��������������������	��&�)�0�2�,�"�	�B�>�?�B�O�V�[�c�[�O�B�B�B�B�B�B�B�B�B�B�����������������������
��������޿*�����	�"�.�G�L�`�y�����������y�m�`�*�"��
���������#�<�H�n�y�w�n�U�H�<�/�"�T�H�>�"����������"�H�a�i�m�z�����z�m�T�������}�z�������������������������������M�K�M�Y�\�f�r�~�y�r�f�Y�M�M�M�M�M�M�M�M�@�=�?�@�@�M�Y�^�Y�P�M�G�@�@�@�@�@�@�@�@�N�M�H�N�R�Z�]�g�k�s�z�s�g�Z�N�N�N�N�N�N�������������¾��������������������������G�;�$� �(�.�T�`�y�����������������y�`�G����������������������$�0�6�<�6���àÓÉ�~�w�u�yÎÓàíù����������ùìà���������z�g�c�d�y���ݿ����������ѿĿ��6����������'�6�B�O�\�p�}��t�g�[�O�6�����������������������������ECE*EEEEEE*ECEPE\EuE�E�E�E�E�EiE\EC�U�P�H�E�H�N�U�X�a�h�a�Z�U�U�U�U�U�U�U�U�����׾ɾ��Ⱦʾ۾�������� ��	�������������������������������������������H�A�?�;�:�;�H�O�T�\�]�T�H�H�H�H�H�H�H�H���e�5�:�H�M�j�������"�,������ֺ����b�`�U�U�J�U�b�n�{ŇŔŠŤŠŗŔŇ�{�n�b���������������������ʼּ̼ۼ�ڼּʼ���������۾Ͼ޾�����.�=�P�P�F�;�����Y�M�@�4�(�4�@�M�Y�f���������������r�Y������������������������� ��������������������ĿľĿ����������������������������FF FF	FFFF$F/F1F=F@F=F=F1F,F$FFF�����������������ùܹ����޹Ϲù������û����������������ûлջܻ����ܻлû_�U�S�R�S�_�l�x�������������x�l�_�_�_�_����������!�-�7�4�-�+�!���������ܹϹù��������ùϹܹ������������������������s�i�k�s�����������������������\�C�6�#������C�O�h�uƈƓƓƎƁ�u�\������������$�*�0�*����������������������������������������������������������}�y���������л�����лû��������������������������������)�1�3�/�&������Ŀ������ѿݿ������@�>�(������ݿľ��������������������������¾ľ����������s�h�o�s���������������s�s�s�s�s�s�s�s�������������������ĽнԽ׽ݽ߽�ݽнĽ�����ݽֽݽ��������(�*�0�(�!�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DӼ��������ּ̼޼������������ּʼ�����¿¦¦¿�������
����	�	����ŠŔŇ�|ŁŇœŔŠŭŹ����������ſŹŭŠ�x�w�x�}�����������ûлһлû����������x�G�F�:�:�:�G�O�S�^�`�c�`�\�S�G�G�G�G�G�G�F�5�!���!�-�:�F�_�l�w�����������l�S�F���ܹ۹׹ܹݹ����	������������h�b�[�Z�[�h�r�t�u�u�t�n�h�h�h�h�h�h�h�h��
����������������������
��#�'�#�����������������ʼּ�����׼ּʼ�����ĘďċďĚĳĿ�������
��&�#��
����ĦĘ��	��������$�+�0�+�$�$�������øàÐÓàìù�����������������ҿпĿ����������������Ŀѿտݿ޿��ݿѿ�������������������������������������������������������������������������� O E W @ ? / Y ) 6 X @ ' R ; @ = d x Q T 9 6 < G #  T < H B G > d q > 3 7 ; > ; ( V Z - ` O C H y 9 r p ]  \ h a H . a 4 L & E T > k { h 6 � A  4    n  �  l  �  �  i  v  �  S    �  �  �  (  j  b  �  H  u  �  V  �  �  �  1  .  D  a  g  �  j    �  K  �  �  '  l  m  �  	  W  :  
  �  n  �    �    L  �    U    v  {  J  Q  �  �  +  �  -  ]  g  b  1  �  <e`B<o;o������C�$�  ��C��t��P�`��9X�T���49X����7L�'�9X�u�u���
��t�����y�#�]/��G���Q��/��G���h�49X��/���ͽ�E���P�C���9X�]/�',1�8Q콓t��H�9�'aG��T���L�ͽ���0 Ž,1��O߽�\)�y�#�,1�,1�m�h�P�`�m�h�]/����u�y�#��o�� Ž�����7L��1��񪽸Q콮{��������9X��jB�_Bk�B,FB'�B �BO�B��B��B�CB x�B� Bu�Bp�BLBC�A���B$��B%��B�B!?�B+��B+�B�B	|B ��A���B�B �BPB�B+[B��Bi[B&�QB�B �A���B'qB��B�B��B �B!�>B@�BM�B��B�8BA�B�rB	[B��B2&�Bb@B$��B&H�B�VB-_�B	\�B�B'�B��B%*B!�B�|BpBy�B
�B��BPBg�B�B$�B�.BDB�6B'	5B @�BT!BB�B>�B*�B.�BBkRB:ZB@B��A��B%{B&6�B�ZB!�]B+@BACB�tBALB ��A���BŋB ��BZB<kBrB�B2�B&��B��B 6
A�w�B@HB��B�AB�B;�B!��BA=B�AB�8B�LBC�B��B	wbB@�B2B;�B$�{B&9CB@B-��B	�B��B'��B��B$��B �B��B�BJ�B
��B?�B�tBB|�B<�A��0A�Y�A�M}A�A��VAA�]A���A��wA<�MA��MA�a�A�WEAiT�A���A�R<A��D@�b@�c�A�!AM70AiɏB�~A�ݒAwUAק�A�'C��NA�u)AW`�A�r0A���@)��A�ƣ@�YEA\<�@���A��JA�K�C���>
�@��3@�kq@a��>�ϐA���BA�i~@���@��VA�CA}�jAM�AEqlA%u	A1�XC�!�A��A���A�&@�
EAl�@�T?&��AۭVA瘺@���A�G�B	OfAϜAyG�A��A/��A�wwA��A�v�A�4�A���AAB'AƁA�jkA<��A�a^A��A�,Ak�A�i�A�|UA�oJ@��J@�40A�I�AMc,AiAB��A�{�Ax�sA� �A���C���A�hEAV�*A��	A���@!u�A�ޗ@���A]Af@�5"A�6A��C��G=��@��v@��@k|�>se�A�{�B�{A��X@��@��A��A|�HAM%uAD�A$�cA3M}C�'�AE"A�{�A���@�� A*�@��K?/�9A�}�A�|�@���A�_B	@�A�u�Ax�_A��`A0��            S            $   0               ;   !                  :   -   &   ]   H      Y               B      	   >               +                     	      #   #                        '      
                  &         -                     C               '            1   !   )                  -   !      -   %      %               =         /                              '   %         #   #   !                     %            %               %      %                     =                           %                                 !   !                     3                                       '            #                           !            %               !      %         N'ӹM��_N6�{P��GN@E�N��NU�8OWA>O,�fO���N6��N��O�H�O�B�O�!�O6��N1\�N��N;T-N�O��1Op'MOh��O�GO��CN�M3ON%PN�?O9�NL�N9yPx�KNػ+NW�yOd�sOQ!�N�YON�(�O23O�c"O!4�N���N��qN#��O�3AO�G�N���NM��O��OkO==zN�zAN(fOU�xN��mN<�XOP�O��O1�N��`N<�
O�'}Ñ2M��&O7�N��DO�gkN�0O͔VO	A�N*��N4�    ^  �  X  �    �  +  �  z  �  �  �  	K  �  P  (  �     j  �  -  �    	�  e  �  �  �  �  �  X    �  8  W  �  �  p  >    �  �  '    �  �  �  �  �  �  �  �  z  �  �  B  �  M    C  �    �  �  	�  �  �  	�  �  �  <��
<T��<T���ě�;��
;�o:�o�49X������`B��`B��`B�D����C���1�T���D���D����o��o�0 ż�h���ͽ<j�C���9X�]/��9X��`B��j��j�o��/��`B�]/�+��h�����t������t��0 ŽC����\)�\)�\)�Y��#�
�t����''0 Ž0 ŽY��D���T���m�h�u�y�#��o��+��C���O߽��P���P������{�� ���������������������� ����������#/3<E</#�#0Un������{b<0���������������������?BIN[[`b[NB:????????66BOU[e[OB<666666666�����
#$(#
�����
#/:<EFA<2/#
\iz�����������zma[[\4<HNSHE<634444444444��������������������&)6BLVS_chqso[B70����������������&)6BO[cekg[OB6)%"& "+/;HLJJJF?;/" #*+# #0<=BA<90(#        {���������yy{{{{{{{{�������������������������������������������������������������������� �����������59BN[������tg[NB:535jnwz�����������zmffj`akmsz��{zmia```````�����
#'' 
���������������������������	%$�������4<HKOKH<494444444444OUaknopnjaUROOOOOOOO|����������������ws|��������������������:<BIUbbb`WUI<;::::::����������������������������������������imxz��������zomfiiii��������������������$/<HKTUWUTPH<<4/,%$$jnz�����������zrlhhj������������������������������������������������������������OO[hkih[OKOOOOOOOOOO��������������������������������������������������������16<BNOYONB6211111111dt�������������ti\_dMQ[gtu�����ztgf[VPNM������	
����������JOT\huw���wuh\OJJJJJ$)67866-)(&#$$$$$$$$�
#/45660#
���.07<?BGILMMJI<40.++."#/'&#������
�������>GR[gt�������tg[NB<>��������������������^bdnsvwyxvnb`YYZ[Z\^���������������������(0<EMMIC;0$
������������������������	����������������������������������������kt������������~yxok���������������������)7DIH@9)
������adnozz���������zniaa )1.)'	��������������	����������������������²¬±²¿��������¿²²²²²²²²²²�H�C�G�H�P�U�_�^�`�U�H�H�H�H�H�H�H�H�H�H���������c�X�C�?�E�T�g�������������������
���������
��� ���
�
�
�
�
�
�
�
�f�a�Z�U�Z�Z�f�s�~�}�u�s�f�f�f�f�f�f�f�f�a�a�V�Y�a�i�n�u�x�u�n�b�a�a�a�a�a�a�a�aùïìäààßàêìù����������������ù�A�=�4�0�0�0�4�5�A�M�Z�_�f�h�j�j�f�Z�O�A�	�������������������	��"�'�/�0�*�"��	�B�>�?�B�O�V�[�c�[�O�B�B�B�B�B�B�B�B�B�B�����������������������
��������޿`�G�:���"�;�G�T�`�d�y�������������m�`�
������
��#�<�H�U�a�n�r�i�U�H�<�/��
�;�0�#����/�;�H�T�\�_�k�p�q�m�a�T�H�;�����������������������������������������M�K�M�Y�\�f�r�~�y�r�f�Y�M�M�M�M�M�M�M�M�@�=�?�@�@�M�Y�^�Y�P�M�G�@�@�@�@�@�@�@�@�Z�P�N�L�N�Z�g�i�s�x�s�g�Z�Z�Z�Z�Z�Z�Z�Z�������������¾��������������������������`�T�E�;�7�5�;�@�G�`�r�������������y�m�`���������������������$�/�4�.�$�!����àÓÎÇÄ�|�{ÅÓàìù��������þùìà�������������������Ŀ߿������ݿѿĿ������ ����6�B�O�d�r�t�j�[�O�B�6������������������������������EuEiE\EPE4E*E$E&E*E6ECEPE\EkEuE{E�E�ExEu�U�P�H�E�H�N�U�X�a�h�a�Z�U�U�U�U�U�U�U�U�׾ѾʾǾʾо������	����	�����������������������������������������������H�A�?�;�:�;�H�O�T�\�]�T�H�H�H�H�H�H�H�H���e�Z�T�Z�a�r�����ֺ��������ֺ����b�a�V�U�L�U�b�n�{ŇŉŔŖŔŇŀ�{�n�b�b�ʼ��������������ʼּ׼ܼؼּʼʼʼʼʼʾ����������	��"�.�3�:�8�0�.�"��	���M�G�@�H�L�R�Y�f�r�����������w�r�f�Y�M������������������������� ��������������ĿĿĿ����������������������ĿĿĿĿĿĿFF FF	FFFF$F/F1F=F@F=F=F1F,F$FFF�����������������ùܹ����ٹϹù����������������������ûлӻܻ����ܻлû��_�W�T�_�l�x�������������x�l�_�_�_�_�_�_����������!�-�-�-�&�!���������������ù¹����ùϹҹ۹ֹϹùùùùùùùùù������������s�i�k�s�����������������������'����*�C�O�h�uƃƐƏƈƁ�u�\�O�C�6�'������������$�*�0�*����������������������������������������������������������}�y���������л�����лû������������������������������������������Ŀ������Ŀɿѿݿ�������������ݿľ��������������������������¾ľ����������s�h�o�s���������������s�s�s�s�s�s�s�s�������������������ĽнԽ׽ݽ߽�ݽнĽ�����ݽֽݽ��������(�*�0�(�!�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DӼ��������ּ̼޼������������ּʼ�����¿±¡¦²¿��������������������ŠŔŇ�|ŁŇœŔŠŭŹ����������ſŹŭŠ�x�w�x�}�����������ûлһлû����������x�G�F�:�:�:�G�O�S�^�`�c�`�\�S�G�G�G�G�G�G�F�5�!���!�-�:�F�_�l�w�����������l�S�F���ܹ۹׹ܹݹ����	������������h�b�[�Z�[�h�r�t�u�u�t�n�h�h�h�h�h�h�h�h�
������������������������
��"�%�#��
���������������ʼּ�����ּռʼ�����ĦĚđčĒĚĳĿ�����
�����
����ĳĦ�����
�����"�$�,�'�$��������øàÐÓàìù�����������������ҿпĿ����������������Ŀѿտݿ޿��ݿѿ�������������������������������������������������������������������������� O E W 9 ? / ` + & Z @ ' I < 7 : d x P T @ , 8 ( '  R < 9 B G 4 R S  4 7 ; > 7 $ V Q 7 ` B C H y . , p ]  \ h a H . a 4 L & E V 8 k o h 6 � A  4    n  �  l  �  �  �  r  �  S    �  �     �  j  b  M  H  s  �  �    �  �  �  .  �  a  g      �  �  �  �  �  '    T  �  �  B  :  �  �  n  �  P  �    L  �    U    �  {  J  Q  �  �  +  n    $  �  b  1  �    F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F      !  '  ,  2  7  :  <  >  ;  2  *      �  �  �  �  �  ^  g  p  y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    (  8  A  F  I  K  H  B  8  +      �  �  �  �  j  H  �     M  X  O  B  <  3    �  �  �  B  �  L  �  I  �  .    �  �  �  �  �  �  �  �  t  U  {  �  [  -  �  �  �  ^  %  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  5  �  �  V    �  �  �      )  $    �  �  �  S    �  �  C  �  a  �  [  H  �  �  �  X  �  �  �  �  �  �  �  �  }  A  �  �       !  y  z  w  q  g  Z  H  4      �  �  �  �  i  U  P  S  i  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  \  H  4      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  q  g  \  �  �  �  �  �  �  �  �  �  �  �  �  b  7  
  �  �  �  [    �  	)  	F  	G  	4  	
  �  �  ~  P  #  �  �  G  �  0  }  �  }  E  �  �  �  �  �  �  �  �  �  �  �  e  %    �  �  �  \    �  M  J  H  J  M  P  P  K  B  .    �  �  �  �  �  l  H  #   �  (  "              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  [  G  1     �   �   �   �   �                    �  �  �  �  �  �  a  =     �   �  j  h  g  e  d  b  a  _  ^  \  Y  S  N  I  D  >  9  4  /  )  1  b  �  �  �  �  �  �  �  �  �  �  �  �  9  �  `  �  �   9  �  �  �    $  ,  *    �  �  �  u  :  �  �    t  �  �  �  \  �  �  �  �  �  �  �  {  F    �  j    �  �  0  �  �  ]  	�  
)  
k  
�  
�  
�     
�  
�  
�  
K  	�  	p  �  2  F  (    w  g  	g  	�  	�  	�  	�  	�  	�  	�  	|  	E  	   �  J  �  [  �    Y  ?  �  e  V  H  :  *      �  �  �  �  �  �  w  f  V  S  T  U  V  	�  
�  I  �  %  n  �  �  �  u  *  �  i      
�  	�  	(  [    �  �  �  �  ~  p  b  T  E  4  "    �  �  �  �  {  J    �  �  �  �  �  �  �  �  �  �  {  `  E  #  �  �  _  �  X  �  <  �  �  �  �  �  �  �  �  �  �  �  {  n  a  S  A  -       �  �  �  �  �  �  �  �  �  �  �  �  �  v  i  ]  Q  D  8  ,    �  "  V  T  <  0  2      �  �  x  6  �  �  #  �  !  �  s  �  �  �  �  �  �  �  �  e  D    �  �  �  ~  U  :  -  �    �  �  �  �  �  �  �  	  �  �  �  �  c  :    �  �  �  Y  )  �  �  %  r  �  �  �    2  5  &    �  �  F  �  P  �    �    :  A  H  T  M  9    �  �  �  {  K  "  �  �  �  f  �  �  �  �  u  h  Z  J  6    �  �  �  i  6    �  �  7  �  �  J  �  �  �  �  �  �  �  {  g  P  5    �  �  �  J  �  �  I  �  p  [  A  %    �  �  �  �  b  @    �  �  �  t  1  �  �      2  <  >  9  1  '    �  �  �  �    O    �  1  �  �  �          �  �  �  �  �  n  H    �  �  z  ?    �  �  �  �  �  �  {  o  _  K  6  !    
    �  �  �  �  ^  +  �  �  �  �  �  �  �  �  �  �  �  `  :    �  �  �  N    Y  }  W  �  �    	    �  �  �  �  �  '  &  %        �  �  �  �    �  �  �  �  �  �  �  �  y  \  K  Q  =    �  �  �  d    f  �  �  �  �  m  P  .    �  �  b    �  V  �  �  q  ^  A  �  �  �  �  �  �  �  �    t  i  _  T  G  :  ,        �  �  �  �  �  �  �  �  ~  q  Q  .    �  �  �  �  `  >    �  �  �  �  �  x  W  #  �  �  U  �  �  x    �  �  P  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  D  �    �  p  =    J  �  �  i  L  ,    �  �  n    �  �  C  �  r  �  >  �  ~  z  v  j  ]  P  ?  +    �  �  �  �  g  @     �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  u  o  j  z  w  l  ]  L  9  $    �  �  �  �  n  B    �  �  e  C  9  �  �  �  �  �  �  �  �  }  e  I  )    �  �  x  1  �  �  3  �  �  �  �  g  A    �  �  �  A  �  �  N  �  �  E  �  �  (  B  :  1  %    �  �  �  �  �  g  L  4    �  �  �  l  /   �  L  x  �  �  �  �  �  j  ;  �  �  �  �  C     �  \  �  N   �  M  E  =  5  )      �  �  �  �  X  I  :  '    �  �  �  U      �  �  �  �  �  �  �  �  �  s  a  N  <  *      .  I  C  0      �  �  �  �  �  �  �  �  �  y  a  J  1    �  �  �  �  y  a  I  (  �  �  �  X  q  �  k  H    �  �    �        �  �  �  �  �  �  �  o  U  9      �  �  �  {  �  �  �  �  �  �  �  �  �  �  �  t  d  U  E  5  &      �  �  �  �  �  �  �  �  �  �  �  �  �  d  )  �  �  /  �  b    �  �  	�  	�  	�  	�  	�  	}  	R  	   �  �  _    �    �  �  f  �  "  "  �  �  �  �  �  �  �  y  ]  >    �  �  a  "  �  y    �  Z  �  �  �  �  �  �  �  �  t  I  #  	       �  �  �  {  V  0  	�  	�  	�  	�  	�  	u  	7  �  �  g    �  e  �  B  h  |  �    �  �  �  �  �  �  g  N  3    �  �  �  �  i  <  "  �  �  �  H  �  �  �  �  �  �  �  �  �  �  �  w  o  e  X  K  ?  2  %      �  �  �  �  �  �  �  �  }  Y  3    �  �  �  p  F     �