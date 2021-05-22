CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�9XbM�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��5   max       P\;%       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       <ě�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?B�\(��   max       @F\(��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @v}�Q�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q`           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�@           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���m   max       <o       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�[6   max       B4�H       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��X   max       B4��       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >!�3   max       C���       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?;   max       C��O       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          O       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��5   max       PF6r       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����   max       ?ө*0U2b       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       <ě�       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?B�\(��   max       @F������     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v}�Q�     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q`           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�I            Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0��   max       ?ӧ��&��     �  ]                                                      8      +                	                              	   
      
         O   H   
   	      
               A         =               
                                 N���N��uO3�hN"CO�~GN�7�N@�)N�B�N�ڛO���OS��N$i�O�qJO6�O���N���N��wP\;%O�q�O�9N��O�H\N��N�_O��O^C�N�4�O9	M��5O^��NtH Nɺ�N��qOuJ�Nd�jN7�3O���N��TN�FNZ�^O�/\O�[IOnyN��N)-�N[�hO�{�NcnO��N#�7P&�O�`O��O���N��TO�,O
O<ٔN��_N��O�_YN�hUN�3O]��O+dON��RO;r�O�m�N��N4i<ě�<e`B<#�
%   ��o���
�t��#�
�49X�D���D���T���T���e`B��o��C���C���t���t���9X��9X��`B��`B��h��h��h��h�����C��C��\)�\)�\)��P��w�,1�,1�0 Ž49X�8Q�8Q�D���D���H�9�L�ͽT���T���Y��]/�]/�ixսm�h�q���q���q���u�y�#�y�#�}�}󶽃o��o��hs��hs����������9X������`B���� �������������	#%"��,/<=HU_bchaUFA<86/*,����������������������������������������EO[\hitutph[UONJEEEENOSUX[dhihfc[ONNNNNN��������������������
#&+010#
���
#/<<4-#
������6?BOY[\bdc`[OB631116~�����������~~~~~~~~#0<IUZ]aXUTC<0"#/<ILKFC?</+(#GITamqz����{maTLHGFG
&)*+)"






:BLO[httuth[OOB7::::#<n{�����{nK<0
{��������������{xwx{")6B[htx�traO6)��������������������:BJNVans����znaUH<7:���������������������������������������������

�����������������������������	
#'#
 ������HTamrz������~zmb[TNH��

�������������Zgt����������tgc[XXZ��������������������-/5<HHPTQRH<;51/----`htz����utshcb``````uv����������������{u��������������������Y[hltwth^[YPYYYYYYYYkovz����������zmfiik��������������������qtx��������������tqqZ[grspg[YUZZZZZZZZZZ6Uakptx{}���nf^U:226������������������������������������z|��������������������')169A6)%$''''''''''����#0AJMH<6110#6<INURJIB<9666666666AHNU]afnz~�|zpnaUH?A��������������������������������������������������������


�������-/3<HUansqrpjaUH</--����������������������������������������������������������������������z��������������zz|zz8;=@GHHNNNJH?;768888N[gt����������tgTNIN���
 
�����������������������������()+)*,--)%���05;BNTYXRNJB=5-,*)+0sty�����������yttss~��������������}{y|~�������������������������������.5@BNZQNB5..........E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���޺ֺкѺֺ�������� ����������������������������������%������������������������������������������������w�f�M�J�F�E�O�Z����������������������w�Y�M�X�Y�d�e�q�r�u�~�����~�t�r�e�Y�Y�Y�Y���	�����������	����������������������������������������������伱�������������������żʼ̼мʼ���������àÛÜÛÕÔßáù������������������ìà�`�T�N�N�`�i�m�y�������������������y�m�`�!���!�'�-�2�:�>�:�-�#�!�!�!�!�!�!�!�!���������������ûл��������ܻлû��)�#�"�%�'�)�6�B�O�[�h�s�n�h�[�V�O�>�6�)��ƳƚƓƗƚƥƳ���������������������������������������������������������������˾��������������������������������������������k�[�C�@�B�f�������������������������ֿĿ������ſѿݿ�����%�(�0�!������"����������*�:�;�H�O�T�a�f�h�f�a�;�/�"�"� �����"�/�/�;�=�B�;�/�"�"�"�"�"�"�s�g�N�5��	���/�A�g�s���������������s���������'�4�@�M�Y�`�Y�M�E�4�.�'����}�x�������������������������������������������������������������������������˻x�b�\�W�S�F�:�5�:�F�_�l�x�|�����������x���������������!�$�-�4�-�!���������������
���#�0�9�<�=�:�0�#��
�������������ɺϺҺɺ������������������������ݾپھ����	���� �����	����ʾʾ��ʾԾ׾ݾ����׾ʾʾʾʾʾʾʾ��
� ���������
��#�/�6�0�/�#���
�
�
�
�������������������������������������������������������������ʾ׾޾���ܾ׾ʾ�ìåãäìóù��������ùìììììììì�����������ùĹ˹ϹйϹù����������������ݿѿĿ������ÿѿݿ�������������FFFFFFF$F1F=F>F=F5F1F,F$FFFFF����ּּ̼ʼǼżʼּ�������������)�(�"�)�5�B�L�H�B�5�)�)�)�)�)�)�)�)�)�)�������������ܺ��'�3�=�'����Ϲ������L�B�E�H�L�Y�r�~�����������������~�e�Y�L�*�(��������� �*�6�C�J�N�I�C�6�*���� ��������������!����������������������������������������������Ҿ������ʾ׾������׾ʾ����������������������������������νݽ���������Ľ������(�4�A�B�A�?�4�(���������Ŀ������������������Ŀѿҿݿ��ݿԿѿ����������������������������������������������!�.�l�y�~����y�m�S�G�!��ܹعչ׹ܹݹ����
����������ܹ�D�D�D�D�D�D�D�EEEE*E.E*EEEEEED��	����ݾξ˾̾վ׾��	�"�@�D�>�4�"��	���������������ûлܻһллû�����������������������'�3�4�8�4�,�'����������������������(�)�(������[�T�J�B�@�B�D�O�[�h�tāČčđĈā�t�h�[�����������������������Ǿʾ׾پ˾ʾ������g�]�Z�N�A�A�A�N�Z�g�s�y�������s�g�g�g�g�I�A�;�3�)�$��"�+�:�I�V�\�m�i�f�b�`�V�I�Ŀ����¿ĿĿѿݿ�����ݿҿѿĿĿĿļ@�?�M�Y�^�f�v�������������������Y�M�@�����������������	��"�/�9�;�>�=�0�"��������������������������������������������/�-�#�����#�/�<�H�U�a�U�R�H�<�<�/�/¦¢ ¦²¿�������������������¿²¦����Ŀĺĸļ����������%�-�2�#�
��������������ÿþ������������� ���������������àÙÓÓÑÓàáåëàààààààààà & P ? _ % \ Z ; 2 I ; I L C & 8 } C ; ? T = h M # � T O ] % 1 A C ? F W > i X + | 5 F m L n k ) : s *  S 9 . 8 . + j | c X � < = D b : Q y  �  �  �  G  �  �  k  �  �  "  �  M  ;  �  {  �  "  �  <  B  �  L  a    C    �  �    �  p  �  �  �  {  N    �  6  c    �  M  :  L  �  �  x  M  j  �  ,  N  �  �  <  [  �  J  �      �  �  z    �  �  	  ~<o�o;D�����
����t��49X�T����t���`B�ě���t��o�o�49X���
���
�����w��o��h�u�,1�'�P�0 Ž+�D���C��e`B��P�<j�0 Ž}�8Q�H�9��hs�P�`�aG��<j���m��h�m�h�e`B�u�u���
�q����hs�e`B����������T��F��7L���㽙����E���\)��7L����������hs��E���vɽ\��9X��h��B�NBS<BO}B�OB"%�BI'B1>B�<B$�WB�yB�IB3�B&!�B>LA��B%�BB&�MB)�BF�B DBBzB �GB,�B/�B!A�B$Q�A�\�B#�"B	��B�tB��B(3B4�HBB��B 6�BwB[KB	(�B�YB�B1�B�oB�
B8?B%�vB&�DBO|BγB��B ��BdBw�BO�B�)BUB7�B=A�[6B
�\Bv�B+SdB�B��B
��B
��B��B={B��B�UBg�B��B�]B"?�BI�B1�BA�B$��B�7B�BD	B%կB#A���B
 B��B&�kB)��B@?B WmBB(B @B?�BEB!F<B$B�A�]LB#�B	��B�_B5�B@�B4��B�1BC^B @BfFB>�B	>3B��B��B
�CB��BC�B@B%`,B&�/B@YB�B=_B �sB5�B>BB;�B$�B��B0BB_�A��XB
��B��B+?�B�xB��B
��B
��B��B7sBhNC�V�@HW	A���A���AD_:?�M�AZh"A�A@�DYA���Am�[@p��@�IaA؝yB!�A��`AJ�IA��A��A�>{A�+A�Gf@ˢ�A��A�L�@���@f��A��c@-AY�FASx�A�j @��AN�rA��>!�3A~�C���A��A��>ok@@A���A�4A��AR��A(A7`�AyadA�
oA�;?'�(C�P�AZ��@���@���A1��A�~8AL��A�D�B��Az�B@��A�ukA���A�(�A���A��AЈ;A�A*C�W�@Dc�A���A�^�AC�?��<AZ��A�~Y@�"A�l7AmE@pc�@��sA���B:RA�WAJp#A�ktA;�A���A��A��O@��A�� A��+@��'@a��A�[�@3#�AZAS��A��@�lAN�A���>?��A~�8C��OApA���>?;@ĖB D�A��AЈ�AS3A#�A7z�Ay#�A�\?A�<?-��C�L�AZq�@�d@�h�A1A�nAI��A��GBC�A{�=@��|A�pA���AÉ�A�y�A�QA�tLA�z�                                                      8      +      !         	                        	      	         
         O   H      	                     A         =   	            
                                                                                       5      '      '            !                                             +                  !            '                              !                     !                                                            1                                                                                       !                                          !                     !      N���N��LO3�hN"CO{�(N�7�N@�)N�B�N�ڛO12@O;�N$i�OGdO6�O��N���N��wPF6rOk�Oj�}N��Ox�N��N�}gO��N�:N�4�O9	M��5O(�NtH Nɺ�N��qO�8Nd�jN7�3OTv�N��TN�FNZ�^O�h�O��OnyN��N)-�N[�hO��.NcnN��#N#�7O��CO�`N��Of�~N��TO�,O
O<ٔN��_N��O�_YN�hUN�3OI@�O+dON��RO;r�O��}N��N4i  f  *  +  *  �  X  j  9  �  �  �  �    s  u     �  �  �  +  �  t  �  �  �  �  E  4  )  �  r  �  :    �  �  �    h  �     
`  q  �  C  �  D  M  �  �  I  �  �  	  �  �  ^  �    :  '  �  �  .  �  �  �  �  Q  <ě�<D��<#�
%   �#�
���
�t��#�
�49X��o�T���T����C��e`B���㼋C���C���9X��1�t���9X����`B����h�+��h������w�C��\)�\)�0 Ž�P��w�<j�,1�0 Ž49X�y�#�L�ͽD���D���H�9�L�ͽY��T���aG��]/���P�ixսu���P�q���q���u�y�#�y�#�}�}󶽃o��o��t���hs����������E�������`B���� �������������!$!�,/<=HU_bchaUFA<86/*,����������������������������������������EO[\hitutph[UONJEEEENOSUX[dhihfc[ONNNNNN��������������������
#&+010#
���
#/030/&#
�����26ABOV[`ba^[OB;63222~�����������~~~~~~~~#0<@IVYZXUI<90##/<ILKFC?</+(#GHKTamz������vmaTNHG
&)*+)"






:BLO[httuth[OOB7::::
#<Un{�����{n<0	
z|���������������{yz).6BOX_df^[OB6)($$&)��������������������FPRUacnz������znaUHF���������������������������������������������

�����������������������������	
#'#
 ������HTamrz������~zmb[TNH��

�������������cgsty��������thg\\cc��������������������-/5<HHPTQRH<;51/----`htz����utshcb``````����������������������������������������Y[hltwth^[YPYYYYYYYYlnrz�����������zmmll��������������������qtx��������������tqqZ[grspg[YUZZZZZZZZZZ7<HUainprrmbaULH<967������������������������������������z|��������������������')169A6)%$''''''''''����#0@IKG<500%#6<INURJIB<9666666666EHQU`acnz|�zznaYUHHE���������������������������
 ���������������������������������

��������8<@HU`ehgfa\UH<52148����������������������������������������������������������������������z��������������zz|zz8;=@GHHNNNJH?;768888N[gt����������tgTNIN���
 
�����������������������������'()+,-)#����05;BNTYXRNJB=5-,*)+0sty�����������yttss~��������������}{y|~�������������������������������.5@BNZQNB5..........E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����ֺҺӺֺ�������������������������������������������%������������������������������������������������f�Z�R�L�N�Z�s���������������������s�f�Y�M�X�Y�d�e�q�r�u�~�����~�t�r�e�Y�Y�Y�Y���	�����������	����������������������������������������������伱�������������������żʼ̼мʼ���������ìáãàÜÙßàìùû��������������ùì�m�b�T�Q�V�`�m�y���������������������y�m�!���!�'�-�2�:�>�:�-�#�!�!�!�!�!�!�!�!�û��������������ûлܻ������ٻл��)�#�"�%�'�)�6�B�O�[�h�s�n�h�[�V�O�>�6�)����ƳƞƖƛƪƳƼ�������������������������������������������������������������˾��������������������������������������������n�^�R�I�H�O�g������������������������ݿĿ������ĿͿѿݿ������$������"�����"�/�;�H�T�V�\�^�^�V�T�H�;�/�"�"� �����"�/�/�;�=�B�;�/�"�"�"�"�"�"�s�N�A�4�(�(�-�6�A�H�N�Z�g�s�y�~��|�v�s���������'�4�@�M�Y�`�Y�M�E�4�.�'����~�z�������������������������������������������������������������������������˻x�u�l�h�`�_�^�W�_�l�x�x�������������x�x���������������!�$�-�4�-�!���������������
���#�0�9�<�=�:�0�#��
�������������ɺϺҺɺ��������������������������������	�������	�����ʾʾ��ʾԾ׾ݾ����׾ʾʾʾʾʾʾʾ��
� ���������
��#�/�6�0�/�#���
�
�
�
�������������������������������������������������������������ƾʾ׾ؾ׾׾Ͼʾ���ìåãäìóù��������ùìììììììì�����������ùĹ˹ϹйϹù�����������������ݿѿÿ����Ŀʿѿݿ������������FFFFFFF$F1F=F>F=F5F1F,F$FFFFF����ּּ̼ʼǼżʼּ�������������)�(�"�)�5�B�L�H�B�5�)�)�)�)�)�)�)�)�)�)�����������������Ϲܹ��������̹ù����e�Y�L�J�K�O�Y�r�~�������������������~�e�*�(��������� �*�6�C�J�N�I�C�6�*���� ��������������!����������������������������������������������Ҿ������ʾ׾������׾ʾ����������������������������������̽ݽ�������ݽĽ������(�4�A�B�A�?�4�(���������Ŀ¿����������������Ŀпѿ޿�ݿۿѿѿ������������������������������������.�!�����	���!�.�G�Y�`�g�j�\�S�G�.�ܹعչ׹ܹݹ����
����������ܹ�D�D�D�D�D�D�D�D�EEEEEEEEEED�D�����ؾؾ����	��#�.�8�5�+�"��	���𻷻������������ûлܻһллû�����������������������'�3�4�8�4�,�'����������������������(�)�(������[�T�J�B�@�B�D�O�[�h�tāČčđĈā�t�h�[�����������������������Ǿʾ׾پ˾ʾ������g�]�Z�N�A�A�A�N�Z�g�s�y�������s�g�g�g�g�I�A�;�3�)�$��"�+�:�I�V�\�m�i�f�b�`�V�I�Ŀ����¿ĿĿѿݿ�����ݿҿѿĿĿĿļ@�?�M�Y�^�f�v�������������������Y�M�@�������������	��"�/�7�;�=�<�/�/�"��	�������������������������������������������/�-�#�����#�/�<�H�U�a�U�R�H�<�<�/�/¦¢ ¦²¿�������������������¿²¦����ĿĻĹĽ����������$�,�0�#�
��������������ÿþ������������� ���������������àÙÓÓÑÓàáåëàààààààààà & A ? _ $ \ Z ; 2 O 5 I C C & 8 } B 3 ! T ' h J # ` T O ] & 1 A C 8 F W 2 i X + F - F m L n k ) 6 s *  ? 1 . 8 . + j | c X � 1 = D b > Q y  �  �  �  G  �  �  k  �  �  �  �  M  �  �  .  �  "  �  �  �  �  �  a  �  C  )  �  �    <  p  �  �  B  {  N  �  �  6  c  9  q  M  :  L  �  n  x     j  L  ,  �  �  �  <  [  �  J  �      �  �  z    �  �  	  ~  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  f  X  G  8  +    �  �  �  �  L    �  �  _    �  l    �      '  )  (  $      	  �  �  �  �  �  �  i  L  *    �  +  "              �  �  �  �  �  �  �    u  j  ^  R  *  .  2  5  7  0  )  "            �  �  �    N  �  �  (  Z  �  �  �  �  �  y  a  B     �  �  �  �  d  '  �  b   v  X  /    �  �  �  \  Z  G  )  �  �  �  `  )  �  �  �  z  V  j  d  ^  X  R  M  G  A  ;  5  .  &              �   �   �  9  8  7  6  4  3  2  1  0  .  -  ,  +  ,  4  ;  B  J  Q  X  �  �  �  �  �  �  �  w  f  R  =  '    �  �  �  �  �  �  �  Q  e  y  �  �  �  �  �  y  e  M  7  #    �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  ^  C  &    �  �  �  �  Z  ?  .  �  �  �  �  �  �  �  �  �  �  �  }  |  ~  �  �  �  �  �  �  �  �          �  �  �  �  �  i  @    �  }  0    �  �  s  s  o  g  W  C  *    �  �  �  o  6  �  �  m  #  �  |  p  j  q  u  o  ^  I  1    �  �  �  r  5  �  �  F  �  f  �  m     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  p  e  Z  N  C  8  ,  �  �  �  �  �  �  �  \  *  �  �  y  :    �  o    �    b  �  �  �  �  �  �  �  e  ?    �  �  }  Q  1    �  �  e  k  M  �  �  �    %  +  )  "    �  �  �  k  '  �  [  �    {  �  �  �  �  �  �  �  �  �  �  x  _  F  ,    �  �  �  �  l    7  <  =  X  m  t  r  j  [  I  2    �  �  V    �  i  [  �  �  �  �  �  �  �  �  s  c  Q  >  4  5  0    �  �  �  �  �  �  �  �  �  �  w  f  Y  R  F  5    �  �  �  �  e  /  �  �  �  �  �  �  �  �  �  �  �  t  e  W  L  A  3  #    �  �  e  o  s  w  �  �  �  �  l  O  5       �  �  �  �  �  �  �  E  :  .  #      	    �  �  �  �  �  �  �  �  �  �  �  �  4  %    �  �  �  �  �  l  D    �  �  �  \  -  �  �  m  �  )  *  +  ,  -  .  /  )  !        �  �  �  �  �  �       �  �  �  �  �  �  �  �  �  �  �  a  6  �  �  d    �  �  �  r  m  g  a  \  V  Q  J  B  :  2  *  "      	     �  �  �  �  �  �  �  �  o  ^  O  E  9  *    �  �  �  �  �  y  �  8  :  1  )           �  �  �  �  �  �  �  �  y  i  Z  L  =  �                        �  �  �  �  9  �    �  }  �  �  �  �  ~  p  e  Y  N  B  4  "    �  �  �  �  �  q  \  �  �  �  �  �  �  �  h  K  -    �  �  �  �  j  K  ,    �  �  �  �  �  �  �  r  S  4    �  �  �  �  H  �  w  �  <  _    �  �  �  �  k  T  =  '    �  �  �  �  |  W  /    �  �  h  T  A  ,      �  �  �  �  k  9    �  �  �  �  m  `  Z  �  �  �  �  �  �  �  �  �  �  {  w  t  q  m  j  g  d  `  ]  
  �  �  �  �  �  �  �  Z    
�  
)  	�  	!  {  �  �    "   �  	�  
9  
^  
]  
V  
I  
2  
  	�  	�  	^  	  �  @  �  �    �  j  �  q  h  `  T  G  9  *        �  �  �  �  �  �  ~  w  X  6  �  �  �  �  �  t  g  [  K  <  #     �  �  �  n  S  >  1  %  C  N  X  c  c  [  U  O  F  >  6  *    @  _  N  <  )    �  �  �  �  �  �  �  z  a  E  #  �  �  �  �  [  ,  �  �    C  6  D  >  )    �  �  �  �  �  �  �  g  H    �  �  n  :  T  M  @  3  %      �  �  �  �  �  �  r  W  9    �  �  y  8  �  �  �  �  �  �  �  �  �  i  F     �  �  }  /  �  �  ;    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  G  �  �    1  E  I  >  '    �  y    �  ,  �     �  �  �  �  �  l  O  F  8  '    �  �  �  �  �  d  4    �  �  v  r  Q  [  �  �  Z  )  �  �  �  >  �  �  U  �  �  8  �  �  F  �  _  �  �  �  	  	  	  	  �  �  �  q  2  �  p  �    I  |  ,  �  �  �  �  �  �  z  k  Y  F  3      �  �  �  ~  D   �   �  �  �  �  �  �  q  ^  J  6  !    �  �  �  �  �  }  g  =  �  ^  Q  A  +    �  �  �  �  �  a  =    �  �  �  <  �  `   �  �  w  ]  ?    �  �  �  z  L    �  �  c    �  s  �  Z  �      �  �  �  �  �  r  V  :      �  �  �  �  }  z  �  �  :  *    
  �  �  �  �  �  �  �  �  �  |  j  W  B  .      '    �  �  �  �  n  J  &    �  �  �  �  [  /  �  �  [   �  �  �  �  �  �  }  a  A     �  �  �  �  c  ;    �  �  �  �  �  �  �  |  y  v  k  Z  I  !  �  �  �  �  p  C     �   �   �  &  .  &    �  �  �  �  k  N  0    �  �  �  V    �  �  �  �  �  �  �  �  �  n  X  @  $    �  �  }  A  �  �  	  u  �  �  �  |  n  ]  H  -    �  �  �  |  N    �  �  ~  l  z  n  �  {  c  L  3      �  �  �  }  S  $  �  �  z  9    �   �  s  �  {  f  N  0    �  �  �  �  |  D    �  �  R  �  .  K  Q  +    �  �  �  a  8    �  �  �  q  C    �  �  R    �    �  �  �  �  �  x  _  J  <  -      �  �  �  M    �  �