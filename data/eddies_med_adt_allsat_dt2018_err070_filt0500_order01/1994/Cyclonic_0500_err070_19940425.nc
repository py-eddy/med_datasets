CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��+J     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MƜ�   max       P�(l     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��E�   max       <�t�     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��
=p�   max       @F�33333     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @v`��
=p     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q@           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��`         0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �   max       ��o     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B 0�   max       B4��     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B 	   max       B4��     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >,�S   max       C��i     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >'�6   max       C���     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          c     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          I     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          A     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MƜ�   max       P�8�     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��2�W��   max       ?�D��*1     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\   max       ;o     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�p��
>   max       @F�33333     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @v`(�\     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q@           �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���         0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?9   max         ?9     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�g�   max       ?�<�쿱\     �  b�   1               
   
      9            A            =   +   b         
                               
               	                                 	   	                  .      O      '         	               	                  (         	OX_N=%<O�N]��N��#N�8�N��O��jP~�7N�/�Nmb�O���P��O�CN$�lP4P2T�P���P�(lN�F�O$��N��rN'$�O�B�N�R4P	��Og��N�_�N#��N�QN_�9OU�.O}ɷO��O�(O�!�N�OIb�O EO7e�N��jOcMƜ�N�m�N�O$p�N2�NHNΟ�N�
�O�lN1M�NuN�N�O�,�N� vP�N�6�P�)O��O�/N�4ZOpN�O�EUO�A�N�qO�?/N��N�O kN���O��Nrg�N��yN��<<�t�:�o%   �o�D����o��o��o��o���
���
���
���
���
�o�t��#�
�#�
�#�
�49X�49X�49X�D���T���e`B�e`B�e`B�u��t����㼬1��1��1��9X��9X�ě�����������/��`B��h��h��h���o�o�o�+�+�+�C��C��\)��P������w�#�
�#�
�0 Ž0 Ž49X�<j�D���D���H�9�H�9�H�9�H�9�L�ͽP�`�Y���%��7L��E���E������������������}�����������������������	
#/4/-)#
 ������������������������+5:BNN[\`^[WNB;5++++$��� #/<BG?</#"        8;BHOU[hjnvyth[L7./8������	������{spt��� 	


����U[_hhtzyutlh[XUUUUUU������������������{��#0m{�����{bI<#����2<IUbn{���rohbUM?872�����������������������5<AHVNB40����_fz�������������ma\_gz����������pfgaz��������������zWa�}tph[RUZ[ehtv~����������	����������������������������������������������������������������"��������������������������./36<HU[hmga^[UH=:/.���������������������'#/5<<</# ��������������������HKJOUanuz�����znaUOH�����������������������������?[gtz�����{wsog[N<:?qz��������������xtnq&).6BLOQTZOBB:6)&&&&CIUbgrsrqnbUIHKLEA?C�������������������� *-6COTTQKC6*!��������������������������������������x�[[chtvyth[[[[[[[[[[[25BINNONB53.22222222��������������������IUaanszz��zna_WUQMI����������������������������������������������������������������������������������������������������������������������������������������������
##(,&#
�����$(*8B[ht}{th[O6)#(05<@DD><0,&# ������������������������������������)Oht���thOB����R[dgt{�������tg][PRR��������������������JN[gltwtsog\[YNLJJJJ�����

�����nt|�����tmnnnnnnnnnn�������������������������������������������������������������������������������������������������y{������{wyyyyyyyyyy���#0<><60/#
����KOR[hmokkhh[UOIHKKKKR[aot{���������tg[Rbhtwyywthcb`bbbbbbbb
#%-)#
./<HLQU[UQH<:/-*....�������������������ûܻ������ܻлû��4�(�(�'�(�4�A�M�R�M�E�A�4�4�4�4�4�4�4�4ìæàßØØÜàçìù��������������ùì���	�	�	���"�*�,�"��������������������{���������������������������~������������������������������H�B�@�A�H�U�a�d�l�d�a�U�H�H�H�H�H�H�H�H�y�m�c�`�X�`�d�t�y�����Ŀ̿ѿͿĿ������y�y�d�W�`�y�������Ŀڿ�����%�#���Ŀ��y�{�v�o�b�V�P�V�b�o�s�{ǔǞǡǨǡǘǔǈ�{�.�'�"�����"�.�0�;�=�;�;�.�.�.�.�.�.���ھʾ����������������׾������	�������a�V�5�7�Q�������������������������������������¼ʼּ������!�����㼽����������������������������������������5�+�,�1�3�?�N�g�s�����������������s�N�5�������ѿ�����*�A�N�]�`�]�P�A�ƚƅ�~�}ƞ������������0�4�<�<�4�����ƚ�����}���~�r�j�k������!�F�b�c�S�!��ɺ��a�a�_�a�c�^�T�H�;�7�/�+�-�/�;�?�H�T�a�a¦²¿����������¿²¦����������������������������������������������������������������ￆ�{�m�\�T�`�m���������ǿȿͿͿ˿Ŀ������N�I�B�=�;�B�D�N�T�[�g�h�r�o�g�[�N�N�N�N���������������6�C�Y�b�h�f�O�6�*�������ʾ��������ʾ׾���������
���{�r�n�b�_�b�n�{ŇŔśŠŢŠŔŇ�{�{�{�{�����������������������������������������6�*�3�6�B�B�C�F�D�B�6�6�6�6�6�6�6�6�6�6ŔŉŔŜŠŭŹź������ŹŭŠŔŔŔŔŔŔ���������������������������������������˾����ݽԽɽʽнսݽ�������(�:�B�4�����׾Ӿ׾����	���"�'�"���	�����)���������������������5�B�L�S�Q�B�5�)�Ŀ����������������ѿݿ����������ѿĻl�j�_�U�V�_�e�l�x���������x�r�l�l�l�l�l�!����!�-�:�S�_�c�n�x���x�l�_�F�:�-�!�ù¹��������������ùϹԹܹ�޹ܹӹϹùþ�ݾ׾;ʾƾ¾Ⱦ׾�������	��������������������������������������������Ň�{�{�n�b�Z�b�e�n�{ŇŔŠťţŠŞŗŔŇ���������������������������������������������*�,�6�8�B�:�6�*���������#�����#�0�;�3�0�#�#�#�#�#�#�#�#�#�#�����������������ùϹܹ���ܹϹ˹ù��������������������������������������������0�*�0�=�I�K�V�X�X�V�I�=�0�0�0�0�0�0�0�0�T�S�J�G�G�G�T�Y�`�c�m�y�z���y�v�m�`�T�T�b�X�Z�b�o�v�{ǆǈǎǊǈ�{�o�b�b�b�b�b�b���v�l�_�S�J�=�F�S�_�l�x����������������������!�-�7�-�%�!���������[�S�O�B�B�B�O�[�h�l�t�v�t�h�[�[�[�[�[�[²­¨¦ ¦¬²¿����������������¿²²�Z�M�A�(�����*�4�A�Z�k�������z�y�s�Z������#�(�4�A�M�Q�T�M�A�<�4�(����4��� �����'�M�r��������������f�Y�4���z�s�g�g�f�g�s�y�������������������������z�p�m�~�|�x�z������������������������������������	��������	������������������	���"�-�'�"���	��������������)�0�6�7�6�.�)�����N�G�B�5�+�)�'�)�4�B�N�[�`�g�p�l�g�[�R�NĚĎčččĚĦĬĭĦĚĚĚĚĚĚĚĚĚĚ�Y�Y�b������ż������	����ּ����p�Y����������Ļļ���������������������Ŀ����������Ŀѿݿ�޿ݿѿοĿĿĿĿĿĺ?�=�@�F�L�Y�c�r���������������~�r�Y�L�?�������������ɺֺ׺ֺܺкɺ��������������'�%�#�'�'�4�7�=�:�4�'�'�'�'�'�'�'�'�'�'�лͻĻ������ûڻܻݻܻۻܻ�����ܻйϹȹù����ùϹܹ�����������ܹϹϹϹ��H�>�/���
���#�<�H�U�\�n�o�a�X�O�K�H�ʼżƼʼּ�������ּʼʼʼʼʼʼʼ�E*E!E'E*E7E?ECEPE\EdE\E\ETEPECE7E*E*E*E*EiEhE_EaEiEuEE�E�E�EE�E�E�E�EuEiEiEiEi H ; H [ P ) ( ; F d R v K " a b @ c : # 7 " U G 3 " C t e n o % p N z . S E , 1 U K k < Q I � W ) I y < i 4 % \ b 0 e N ! N 2 < � 8 X > @ J Q . ; H Y N  �  \  b  �  �  �  �  �  s  )  �  �  /  �  a        �     g  �  e  
  �  X  �  �  J  t  �  �  V  S  T  �  �  �    �  �  :  ,  �  (  ^  �  w  �  �  �  M  �    $    �  �  ,  ^    �  *    
  �  �  9  �  <  �  �  /  �    ���󶻃o�D�����
�o�T���e`B��j�u�ě��t����ͽ�O߽+�49X����\)�Y���/��o����1��j�t���`B�8Q��󶼋C���9X��/���ͼ��D���+�D���P�`�C��H�9�49X�@��+�C����\)�\)�]/�t��''8Q�L�ͽ��'�+����T����<j���w�u�e`B�T���m�h�L�ͽ��-���P�ixս�%�y�#�Y���o��o��񪽟�w��h�ȴ9B��B�;B&?B)~B;B�JB7B�B+B��B�tB gmB&Z�B'TMBh;B#/B 0�B?�B�
B�B�9B4��B��B:BtpB]BA�B�BEVB��B�7B��B!�.B\�B	%+BJB�B'o�B�9B07�Bc�B/XB͝B� B4�BzaB��B
�B�BIB!�B"��B/�B��B	B%�6B�B��B"3B	�-B�_B�BB��B
'B,��Bg�B�?B!:B!*JB)5B$�BNaB
N<B�B�BķBKEB�VB<TBOvB�BB�B?B<%B*�NB�B�B �sB&�fB'?VB@8BH�B 	B��B>pB�rB�B4��BDB�BB�B��BA�BǮB�B��BǠB7jB!�yB>�B��B@`B?�B'�{B�B0Q�B�B0BĢB�B?�BAjB�'B
�B�B>cB"A�B"ΒB?�B?lB4sB%�dB��B-&B��B	AYB5]B�!BWOB	߾B,ÝB��B �B!P�B!;<B(��B$�eBF<B
D3BB=�B<2@�6�A:cA�AOA��XAHYNAH�"A�;"ArUmAx$�B��A_�iAR��A���AK_A�!lA�\�A�V�B��@G�A��0A�>�AK�A[kAq0�A�AA�V@AUS�A� �A���A���A�YeA�.vA0�AYZA�(uA|ʭ@���@|�>,�SAU��A�CoA�8�@���A�xiA��D>8j�A�jyB
��Ai`&B��@�s�@j	�AڞoA��MA=݆A7߮@�&%A���A��8AZ�MA[XA՝�A�,3A��+@�,TA��Az\"?��@/�@�{U@�+�>��A�A�A�C���C��i@��A:.^À�A��bAH�AG�A�~�At�/Avh�B��A`��AP��A�~ A
�A��KA���A���B�@K�LA�HA��9AK��A JAs�A�q�A���AR�A�pjA�|AׁPA�r�A���A-�AY/A��4A}�@�F�@t-�>C�rAU��A�A���@���A��
A��>'�6A�y�B
�ZAi8B@�@���@o�A��A��IA>�1A8�@��<A�8A�lzAZ��AZ�A�Y�A��A���A7�A�Az�?���@/�S@��O@�N�>���A�~VA�C��xC���   2               
         9            A            =   ,   c                                  	                     	                                 	   	                  .      O      '         	               	                  )         
                        !   9         !   ?         )   -   =   I                     '         
            !      %                                                            %      /      )                  /   #                                                         1            =         )      7   A                     #         
                  %                                                                  )      '                  /                                 N��N=%<O�N]��Nn@�N�8�N'(�O�=�P"T�N�P�Nmb�O\$�P���O�SjN$�lP4Op�PS\hP�8�N�F�O$��N��rN'$�O�B�N�R4O�ilOg��N�_�N#��N�QN_�9O3)mO�0O��O��Ony�N�N�=IN��~O7e�N��jOcMƜ�N�m�N�OɂN2�NHNΟ�N�
�O�lN1M�NuN�N{�1OZ��N� vO�L�N�6�O��%O��O�/N�4ZOpN�O�EUOd}�N�qO�?/N=Y�N�O9�N)�xO
�!N[�)N�,NP�  	�  l  7  ^    �  n  q  3  k  �    �    �  �  �  �  �  �  �  �  �  �    o  �  [  k  @  }  (  x  ,  �  �  �    �  �  �  <      �  �    �  �  )  �  �  �    �  �  _  �  �  �  �  �  S  �  �  �  	    N  �  �  �  S    	�  #;o:�o%   �o��o��o��`B�ě�����o���
��`B�o�#�
�o�t���P��t���j�49X�49X�49X�D���T���e`B��t��e`B�u��t����㼬1��9X��`B��9X��j�������+��`B��`B��h��h��h���o�C��o�+�+�+�C��C��\)�<j�aG����'#�
�49X�0 Ž0 Ž49X�<j�D���D���Y��H�9�H�9�T���L�ͽT���e`B������C��\��Q�������������������������������������������	
#/4/-)#
 ������������������������=BN[\[ZONB@5========$���!#/<<<<5/)#!!!!!!!!!27?BO[hjltvth[NB;112{����������������}z{���

������U[_hhtzyutlh[XUUUUUU���������������������#0In{�����{bI0����8<IUbny|zunibUF>::68�����������������������5<AHVNB40����jmz�����������zmhfgjpz�������	�����snpn���������������zfbn�}tph[RUZ[ehtv~����������	����������������������������������������������������������������"���������������������������./36<HU[hmga^[UH=:/.���������������������'#/5<<</# ��������������������PUanpz���}zngaUTJMKP�����������������������������?[gt|�����zvrng[N=;?}��������������~zxv}&).6BLOQTZOBB:6)&&&&EIKUbbnnnnkhbUSLIFEE�������������������� *-6COTTQKC6*!��������������������������������������x�[[chtvyth[[[[[[[[[[[25BINNONB53.22222222��������������������MUanqyz~��zna`XURNMM����������������������������������������������������������������������������������������������������������������������������������������������
#$&#
�������BBO[hmstrnhg[OC:79=B#(05<@DD><0,&# ���������������������������������)Oht}��th[OB�R[dgt{�������tg][PRR��������������������JN[gltwtsog\[YNLJJJJ�����

�����nt|�����tmnnnnnnnnnn�������������������������������������������������������������������������������������������������y{������{wyyyyyyyyyy���
!#.030.#
����NOV[hklhhe[UONNNNNNNot~����������}tnhjoochtvyxvthdbacccccccc
"#&#
	+/2<HKPH<7/-++++++++�û��������������ûƻлܻ߻���ܻٻлþ4�(�(�'�(�4�A�M�R�M�E�A�4�4�4�4�4�4�4�4ìæàßØØÜàçìù��������������ùì���	�	�	���"�*�,�"���������������������������������������������������~������������������������������H�E�D�H�H�U�V�a�e�a�[�U�H�H�H�H�H�H�H�H�����m�h�b�i�x�������ĿǿϿ˿Ŀ����������������u�n�p�������ĿϿտڿ�� ��
���꿸ǈ�~�{�o�b�o�w�{ǈǔǗǡǥǡǔǓǈǈǈǈ�.�'�"�����"�.�0�;�=�;�;�.�.�.�.�.�.��ʾ����������������׾�����
�	����������d�Y�K�;�B�U�s������������� ���������������������ʼּ������������ʼ�����������������������������������������5�+�,�1�3�?�N�g�s�����������������s�N�5������ �����(�5�A�G�M�N�H�A�5�(�ƚƐƍƚ�������������$�2�8�4�$�����ƚ�����������y�r�������:�Q�\�W�!��ɺ����a�a�_�a�c�^�T�H�;�7�/�+�-�/�;�?�H�T�a�a¦²¿����������¿²¦����������������������������������������������������������������ￆ�{�m�\�T�`�m���������ǿȿͿͿ˿Ŀ������N�I�B�=�;�B�D�N�T�[�g�h�r�o�g�[�N�N�N�N������������*�6�C�U�_�c�^�O�6�*�������ʾ��������ʾ׾���������
���{�r�n�b�_�b�n�{ŇŔśŠŢŠŔŇ�{�{�{�{�����������������������������������������6�*�3�6�B�B�C�F�D�B�6�6�6�6�6�6�6�6�6�6ŔŉŔŜŠŭŹź������ŹŭŠŔŔŔŔŔŔ��������������������������������������������ݽڽϽнҽݽ������#�(�)���������׾Ӿ׾����	���"�'�"���	�����)���������������������5�B�K�R�P�B�5�)�Ŀ����������Ŀѿݿ�� ��������ݿѿĻl�j�_�U�V�_�e�l�x���������x�r�l�l�l�l�l�-�#�!�� �!�-�-�:�;�F�S�[�Z�S�F�A�:�-�-�Ϲɹù��������������ùϹӹܹ߹޹ܹѹϹϾ�ݾ׾;ʾƾ¾Ⱦ׾�������	��������������������������������������������Ň�{�{�n�b�Z�b�e�n�{ŇŔŠťţŠŞŗŔŇ���������������������������������������������*�,�6�8�B�:�6�*���������#�����#�0�;�3�0�#�#�#�#�#�#�#�#�#�#���������������ùϹܹ��޹ܹϹɹù����������������������������������������������0�*�0�=�I�K�V�X�X�V�I�=�0�0�0�0�0�0�0�0�T�S�J�G�G�G�T�Y�`�c�m�y�z���y�v�m�`�T�T�b�X�Z�b�o�v�{ǆǈǎǊǈ�{�o�b�b�b�b�b�b���v�l�_�S�J�=�F�S�_�l�x����������������������!�-�7�-�%�!���������[�S�O�B�B�B�O�[�h�l�t�v�t�h�[�[�[�[�[�[¿µ²®°²º¿������������¿¿¿¿¿¿�4�4�+�+�0�4�A�M�Z�f�r�s�~�{�s�f�Z�M�A�4������#�(�4�A�M�Q�T�M�A�<�4�(����4���������'�4�M�r�����������r�Y�4���z�s�g�g�f�g�s�y�������������������������s�o�o�����}��������������������������������������	��������	������������������	���"�-�'�"���	��������������)�0�6�7�6�.�)�����N�G�B�5�+�)�'�)�4�B�N�[�`�g�p�l�g�[�R�NĚĎčččĚĦĬĭĦĚĚĚĚĚĚĚĚĚĚ�Y�Y�b������ż������	����ּ����p�Y�����������������������������
������Ŀ����������Ŀѿݿ�޿ݿѿοĿĿĿĿĿĺ?�=�@�F�L�Y�c�r���������������~�r�Y�L�?�������������ƺɺֺ̺ٺֺ̺ɺ������������'�%�#�'�'�4�7�=�:�4�'�'�'�'�'�'�'�'�'�'�лϻŻ����������û˻лջܻ������ܻйϹ͹ù����ùϹܹ޹ܹ۹ܹ߹ܹϹϹϹϹϹ��#�����#�#�/�<�H�U�]�W�U�M�H�<�/�#�#�ʼƼǼʼּ�������ּʼʼʼʼʼʼʼ�E*E$E*E*E7ECEFEPEVEYEQEPECE7E*E*E*E*E*E*EuEkEiEbEcEiEuE~EE�E�E�EuEuEuEuEuEuEuEu + ; H [ > ) . ; C J R p H  a b   n E # 7 " U G 3 % C t e n o " a N z   S - * 1 U K k < Q F � W ) I y < i V  \ U 0 a N ! N 2 < � $ X > 0 J D A 0 J S I    \  b  �  p  �  F  K    �  �  E  �    a    �  W  �     g  �  e  
  �  �  �  �  J  t  �  z  �  S  =  �  �  �    �  �  :  ,  �  (  D  �  w  �  �  �  M  �  �  �    �  �  �  ^    �  *    
  �  �  9  T  <  Z  I  7  y  �  n  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  ?9  �  �  	t  	�  	�  	�  	�  	�  	�  	k  	(  �  �  9  �  _  �  h  �    l  h  e  a  ]  W  P  I  @  4  (      �  �  �  �  r  M  (  7  1  )        �  �  �  �  �  �  �  }  f  G    �  6  �  ^  \  Z  X  W  U  S  O  J  E  @  <  7  2  /  ,  )  &  "                "  %  (  *  ,  .  .  .  .  -  '         �  �  �  �  �  �  �  z  g  S  ?  *    �  �  �  �  �  �  �  E  O  Z  b  i  l  m  i  b  X  O  E  9  +      �  �  �  �  a  l  q  p  l  b  O  ;  %    �  �  �  �  O    �  �  �  `  �  �    #  /  3  )    �  �  �  �  Z    �  o    �  �   �  '  [  c  k  U  8  
  �  �  I    �  x  %  �  x  !  �  m    �  �  �  �  �  �  q  ^  K  9  '       �   �   �   �   �   �   �  �    
    �  �  �  �  �  �  �  �  b  7  �  �  i  6  0  t  �  �  �    J    �  a    �  M  �  �  1  �  �  �  4  �    �          �  �  �  �  �  �  �  r  G    �  d  �  7   j  �  �  �  �  �  �  �  �  �  �  �  y  r  i  _  T  I  ?  4  *  �  �  �  �  ~  d  D    �  �  �  �  �  f  S  <    �  �   �  �  Q  r  �    n  �  �  �  �  �  v  C    �  N  �  �  '  >  O  f  n  z  �  k  M  A  9    �  �  :  �  V  �  �  g  #  O  �  �  �  �  �  s  4  �      �  �  �  T  �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  Y  ?  $    �  �  �  z  S  $  �  x    �  G  �  �  �  �  �  �  �  |  n  [  F  0  !      �  �  �  �  �  �  �  �  �  }  g  N  4    �  �  �  �  y  \  C  [  �  �  �  �  �  �  {  j  W  A  %    �  �  �  O    �  �  |  !  �   �          �  �  �  �  �  r  A    �  �  �  Q    �  n    d  i  n  l  f  Y  K  9  #    �  �  �    ^  ,  �  �  �    �  �  �  �  �  �  �  �  g  L  /    �  �  �  �  �  Y  !  �  [  \  ]  ]  ^  _  _  `  `  a  _  Y  T  N  H  C  =  8  2  -  k  g  d  `  \  X  S  O  K  F  B  =  9  4  0  T  �  �  �    @  5  )      �  �  �  �  �  U  +    �  �  |  N     �  �  }  w  q  j  d  _  ^  ]  \  [  \  `  c  g  j  s  }  �  �  �  '  (  (  $       	  �  �  �  �  �  �  �  �  �  �  �     �  <  Q  \  a  k  x  s  c  L  3    �  �  s  -  �  �  #  o  �  ,  %        �  �  �  �  �  [  1    �  �  �  W  $  �  �  �  �  �  �  �  e  4  �  �  {  S      �  �  h    �  �  �  -  \  x  �  �  �  �  |  d  B    �  �  �  �  V  	  �  )  �  �  �  �  �  w  b  L  5      �  �  �  �  �  �  �  �  �  j  �  �  �  �  �  �    �  �  �  �  y  H    �  �  K    �  ~  �  �  �  �  �  �  �  �  q  L  &    �  �  �  Z  )  �  ~    �  �  �  �  �  �  �  �  y  _  C  $    �  �  �  V    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  k  Y  G  <  7  1  ,  &          �  �  �  �  �  �  �  �  �  �  q           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {      �  �  �  �  �  �  �  �  �  �  �  t  ]  E  ,     �   �  �  �  �  �  �  �  �  �  |  p  c  W  J  >  2  &        �  �  �  �  �  �  �  �  �  ~  R    �  �  =  �  �  5  �  �  P            �  �  �  �  �  �  �  z  r  k  R  5     �   �  �  �  �  �  �  �  �  |  j  W  D  /       �  �  �  o  <  
  �  �  �  �  �  �  �  �  �  �  |  a  F  '    �  �  �  �  y  )  &  #        �  �  �  �  `  3  �  �  g  
  �  A  �  k  �  �  w  h  e  f  j  �  �  �  �  �  �  �  o  N  ,    �  �  �  �  �  �  �  �  �  �  �  �  u  U  5    �  �  �  �  �  s  �  �  u  ]  D  *    �  �  �  �  m  8    �  �  �  g  =    =  �  �  �  �  �    �  �  �  H  �  .  �  ]    �  M  �  �  '  =  I  Y  h  �  �  �  �  �  v  T    �  �  I  �  �  �  �  �  �  �  q  W  ;    �  �  �  �  �  \  .  �  �  Y    �  ~  �  Z  D  !  �  �  r    
�  
*  	�  �  2  s  �  >  �    �  �  �  �  �  �  �  �  y  d  P  ;        �  �  �  �  k  I  '    �  �  �  �  �  �  �  g  R  5    �  �  _     �  A  �  6   �  �  �  �  �  �  �  �  q  O  +    �  �  j  <  !    �  �  �  �  �  �  �  �  �  �  o  ]  J  6  "    �  �  �  t  )  �  �  �  �  �  }  y  s  h  ]  Q  E  8  +      �  �    �  �  �  S  G  9  *      �  �  �  �  d  ;    �  �  z  D  +  )  7  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \  �  |  l  U  .    �  �  �  l  6    �  �  I  �    �  z    �  �  �  z  q  n  k  h  a  S  D    �  �  S  
  �  	  �  �  �  �  �  �  �  j  O  3    �  �  �  �  v  Y  =  "      �  �  �  �  �  �  f  E  #  �  �  �  y  :  �  �  �  �  3  :  A  H  L  K  ?  1       �  �  �  �  �  w  O  �  �  \  �  |  w  q  l  g  a  e  l  s  {  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  R  5    �  �  �  �  q  �  �    �  �  �  �  �  �  �         �  �  �  �  �  �  �  ~  d  I    u  �    1  M  Q  D  4    �  �  �  L  �  �    r  G  �      
  �  �  �  �  �  �  �  u  i  `  S  G  A  9  0  %    	q  	r  	x  	�  	�  	�  	l  	<  		  �  �  Q  	  �  p    F  y  �  �  �    !          �  �  �  �  �  p  P  /    �  �  �  �