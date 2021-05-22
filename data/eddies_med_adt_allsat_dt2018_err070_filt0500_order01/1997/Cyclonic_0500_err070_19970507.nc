CDF       
      obs    R   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�t�j~��     H  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�X   max       P��     H  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��hs   max       =+     H   <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��
=p�   max       @F�(�\     �  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���\)    max       @vr�Q�     �  .T   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @P�           �  ;$   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�V@         H  ;�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �      max       <�h     H  =   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��6   max       B1�T     H  >X   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}>   max       B1��     H  ?�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >]�=   max       C�Y�     H  @�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C��     H  B0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          M     H  Cx   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7     H  D�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7     H  F   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�X   max       P�bs     H  GP   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?�o hۋ�     H  H�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���-   max       =+     H  I�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��\)   max       @Fk��Q�     �  K(   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���\)    max       @vp�����     �  W�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           �  d�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�-�         H  el   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�     H  f�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��$�/   max       ?�jOv`     �  g�      	   	                     	                  6   1                                                4      %         $   
                  &                     	         0   L   5      !   0         :         :               L      	                     )      N0ɿN�0�N�#�Np6Oz�nN��Nr�O&S�O�6~N�C�N��N�cN�XO��OD�P��O�D�N���O:�O��N�L�NNf|dOw��N�o�N��@Nyn�NK'mN�~N�}}O!�*O,P'O�HODtO���NU�NNH �O�2�O:��N�UlN�m�N�r�OU�NėHO�C<O��YOd�O�3�Oy�N1�FNY2�N��-NW��O��P~MP�UP��N��O9�PN�N�O/"&P3_O��N$�O�*�O�XO4�O��aN�d*O�.N��@NS��O���OmNi��O��OMFwN�y�O�s�N��4No��=+<��<ě�<T��<t�;�`B;ě�;��
;��
��o�o�o�#�
�D���D���D���T���T���u�u��o��C���t����㼬1��1��1��1��1��j�ě��ě����ͼ�����/��/��/��/��`B��`B��h��h��h���������o�+�+�+�+�+�\)�\)�t��t���P��P��P������w��w�#�
�#�
�#�
�49X�<j�<j�H�9�H�9�H�9�Y��]/�e`B�e`B�ixսixսu��%��hs%����������������Z[hmty��ythh[RZZZZZZ
 !
<BN[gotyyyxwtg[ND?=<>IUXbnnbaUNIF@>>>>>>vz������zyvvvvvvvvvvBHSUamnz���}znaUHC=B9C\hu���������uhOC=9����


����������?BIN[acec[NMBA>=????"#./149/*#"""""""""��������������������]ahmz��������zue^\]��������������������#0U������{b<.������
#,("
�����"+/<HU_YUTHG=<510/#"��������������������������������������������������������������������������������!$$)*)�����������������st|������������}ttss��������������������������������������������������������������������������������EO[hmtwwth[ZOLEEEEEE���#�����������
�����������
#/HQUUH</#
������nt|���������������wn��������������������������������������������������������������������������������������������������������������������jmpwz���zzomijjjjjj)5>BEEEB54)&")*06:BO[^`\[VOB6+)"��������������������#/<HYdjmlaH</#������ ������)5BGN[gt�����t[B5)p���������������tllp_abmz��������zpmda__������������������������������������������������������������W[gnsqg[XSWWWWWWWWWW[nt�������������tc[[����������������������������������������%)6O[t��}thOB6)��������������������#,/5<?BDDA></#" #/Ha��������oga\Q<2,/�����������������������������������������������$.1+%������T[gt���������tg[Y\TT
#$#
/3<HUansttnaUOH</))/9BL[hhqtuuuth[VOIDB9>BN[\egjllgc[NHB=8<>����
#0:?BCB<0#���:<ILQUXYYWUIA<<;::::)5BNfa[PB5)�����������������������������������������������  �������
"#*.#
�������
#&#

������FKUbn{�������tlUI?AF55BN[_fgqrgfNBA95325v�������������vvvvvv���"$$�����������������������������������������������@�>�3�0�3�<�@�K�L�S�Q�L�@�@�@�@�@�@�@�@¦¦²²³¿��������¿²¦¦¦¦���	�����!�-�1�-�-�$�!�������������������ɼʼҼʼ������������������������{�z��������������Ŀпѿ˿Ŀ��������ʼ����������ʼּּ߼���ּʼʼʼʼʼ����������������������������������������ҾA�;�4�3�2�4�5�:�A�M�Q�Z�]�c�d�d�]�Z�M�A���������������������ʾ̾ξӾ־־ھӾ���àÙààäìù��������������ùìàààà�A�<�5�0�,�5�A�N�Z�`�g�g�h�g�Z�N�A�A�A�A�<�;�/�&�/�<�H�T�U�U�U�H�<�<�<�<�<�<�<�<�;�8�:�;�H�T�[�T�M�H�;�;�;�;�;�;�;�;�;�;���������������������$�,�+�'�$�������������������
��#�/�9�<�H�Q�H�<�/�#�
�����s�\�W�R�A���������������������������������s�q�|�������������ݿ�����ѿĿ����B�6�5�1�1�,�6�B�I�O�X�[�h�i�h�[�O�N�C�B��żŹųŷŹ���������������������ƿ	������������	��"�.�;�I�I�;�.�"��	��������������������������������������������#�"�������������x�v�l�i�l�x�����������������x�x�x�x�x�x��Ƶƾ���������$�0�1�*�$�������������T�S�H�G�H�N�P�T�Y�a�m�w�r�p�p�m�a�Y�T�T�z�y�t�q�y�z�����������������z�z�z�z�z�zŭŤťŧŭŷŹź��������Źůŭŭŭŭŭŭ�`�`�T�L�S�T�U�`�m�w�y�������y�m�`�`�`�`�����������������������������ĿſпĿ����������������ûлһӻһлŻû������������{�z�u�y�{ŃŇōŔŘŠŭŰųŭūŠŔŇ�{�A�5�.�'�)�5�N�[�g�f�s�t�f�e�]�Z�T�W�N�A�����������������������!�*�1�*�����ݻ��������������z�~�����������������������O�B�@�H�[�hāēĚĦīĲĮĦĚčā�t�h�OD�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�D��z�t�n�c�a�\�a�n�w�zÇÈÇ�}�z�z�z�z�z�zÓË�z�n�f�^�[�a�n�zÓàäùþüùìàÓƎƁ�u�h�e�h�f�h�uƁƎƚƝƥƧƨƧơƚƎ�C�A�7�6�1�6�?�C�E�O�\�b�h�j�h�\�V�O�C�CìãàÓÎÓàìù��������ùìììììì�����������������������������������������r�h�f�Y�Q�M�C�B�K�M�Y�f�g�r�������}�r���������������������������������������侮�������Ǿ׾����	������	���׾��������׾Ӿо׾���	���.�5�=�?�;�.��	�������������	��������	��ĻĳĹĿ��������������(������������������������������	���!���	�����������������!������������5�1�1�/�5�A�N�S�R�N�A�5�5�5�5�5�5�5�5�5�"��"�%�"��"�/�;�H�K�T�U�T�H�E�;�/�"�"�6�/�3�6�C�O�U�S�O�C�6�6�6�6�6�6�6�6�6�6�{�n�h�U�I�>�6�9�I�U�{ŇřŠţũŤŔŇ�{�����~�n�h�n�~�����ֺ���!�*������ɺ��p�X�T�`�o�����������ûлܻ�����������p�����w�m�n�l�n�t�������������������������b�a�V�V�U�N�V�b�j�o�{�~ǀ�{�p�o�b�b�b�bE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��l�b�t�����ùܹ���4�/������Ϲ������l�ʼü��ʼּ����ּʼʼʼʼʼʼʼʼʼʼ@�4�'������'�4�@�M�Y�\�g�h�_�Y�M�@������ĿĨč�b�W�[�hāĚĦĮ������������	��������������,�5�B�F�W�\�N�B�)�EPEJECE@ECEMEPEQE\E`EdE\EPEPEPEPEPEPEPEP�2�(����#�5�A�N�Z�y�����������x�g�N�2�Ľ����������Ľνнݽ�������ݽнɽ��)��"�(�)�6�;�B�O�[�e�h�l�j�h�c�[�O�6�)�нɽ������������������Ľн�����ݽо���������(�4�A�D�K�A�4�4�(����óäÚÖ×ßù�������	��� ����������ó�6�6�6�<�<�C�G�O�U�\�b�\�V�O�N�C�6�6�6�6��|�r�r�r�y������������������������ݽн½ŽϽݽ߽�����(�4�8�/�#�������� �����!�-�8�:�:�<�<�:�-���������� �����������������ܻ̻ȻȻ˻ջ���@�N�M�@�4�������#�"�����#�$�/�<�F�H�^�W�U�H�B�<�/�#�<�5�/�.�/�1�<�H�U�W�W�U�H�B�<�<�<�<�<�<�.�����޼�����.�:�`�l�x�w�`�R�G�:�.ììà×ÓÇÁÇÓàìôùúùîìììì�������������������������������������� @ L < 4 + ^ J S w 4 0 S N A c & A O S " O ? O p W M E � N e 0 ^ g P 0 i Q B % 6 d C : 7 7 Q w 4 # X G D # G P V . : N a N T � S J 5 4 , . ` ' X j G < i E " H Z [ 9  H  �  �  x  �  �  N  �  �    �  G  6  %  �  _  A    �    �  <  u  w    �  �  �  �  �  e  t  �  �  h  �  p  y  �  �  �  �  X  �    l  r  �  9  W  �  �  a  �    �  �  �  r  �  &  �  �  �  7  �  D  �  [       �  �  �  M  �  L  �  �  -  �  }<�h<�j<u:�o�u;o;D���49X�u�#�
��`B�e`B��o�C��t���+�}���ͼ�9X��P��j�ě��ě�����/��/�����ě��ě��C��49X��󶽝�-�'�o�8Q�,1��%���\)�#�
��P�ixս0 Ž�7L�Y��D���]/�'\)���''y�#���`B��E��0 Ž�O߽�1�8Q콁%�ě��m�h�Y��Ƨ�T����+�����y�#�   �Y��ixս�hs��C��m�h��{���\)������-����BKiB�B��B${�B��B'.�B��BX�B1�TB�B]�B3EB �A��gB��B'�B�B�B�B��B!`�B�B��B%�B
�xB��BrB+�TB+�)BR�B��B��B�B�#B�5B�BX�B?�B�;B��A���B�BB�Bp�B�B��B	+�B
�|A��6B1�B޲B)	B	�B
��Bq�B^�B��B��BL�B�WB�`B �+B -B
\�B0�B}BJ.Bh)B$�B&�hB�BvBT2BE>B$PsB$e�B(��BA�B
̰B�aB
��B?=BC�B��B�B$@NB�B'�B��B�B1��B��B[�B�B 2�B bB�PB&�ZB�BB�B��B�-B!��B�qB��B��B
�3B=�BM�B+�.B+ҤB��B�OB]~B;�B�B��B@!B��B�|B��B!�A�}>B=�BD�BѺB�B��B	AB
@B ;TB?B9lBE�B��B
R�B��B��B��B=BB#BáB�B ��B��B9�B>�BRBFaB�@B%?�B&��B�B��B�8B�UB#��B$�)B)>wB7�B
�$B��BS�B?�?�r[A�{D@h�i@�IdAs�L@��[A��HA<�bAM�~Á�A�efA�bA�|$BuKA���A�1'Av�eA��A��A^%AJP�AԡH@�w.B�A�d�A�tA��Aj�Au�@��'A�]=A��A�,�@�gAܥ(C�8�A��_A�%�B]�B2�Ä}A�̻@ݥzA�ڜAU7JAZ׋AY�A�*-A��.A�
%A�CA�[B �_A�}@/ �@��A�Bl�C�Y�>]�=A �-@���A�	tA�}�C���A�LA)$�A���A&�A5�/A�n�B �9@�}�A0��@j�@���@�7�A��tA���A
��A�x�B�H?���A�F@d[L@��As�^@��BA�Z|A<7�AMtA�^�A�Q�A�{�A��pB��A��VA��JAv�CA؀=A�u�A^�AI]A�{�@��,BOA��A��UA���Ag.}At��@� nA�|�A�ÐA�hL@�� A܀C�;�A�}tA�~�B��B'�A�j_A�gJ@�ӇA��7AV��A[��AYG�A�~�A��=A�zA��A��8B �IA�U@#�@�
/A���B�cC�Q�C��A �B@���AߋA���C���A�t�A(��Aن�A$��A6��A�wgBP�@� �A1p@d��@�(@��BA�AÁ+A
�zA�޻B��      	   
                     
                  7   2                                                5      &         %            	         &            	         	   	      0   M   6      !   0         :         :               M      	                     *                                                      7   %                                             #                                       !         %                     -   -   '         7         7   %      #               !         #         '         '                                                      5                                                                                       !                              )   )            7         7                                 #         '               N0ɿN�0�NlZ�Np6O$�N��Nr�O&S�O�N�XTN��N�cN�XOd�OݽP�bsO��N��O:�O)EgN�L�NNf|dOfa�N�o�N��@Nyn�NK'mN�~N�}}N���Ng[�N�SyObOxNU�NN0��O�$eO:��N�ƠN�m�N�r�O�N�ӨO��}O���O��O��N�N1�FNY2�N��-NW��O�8cP��O��WO��N��Nڢ�PN�N�O/"&P(Y$O���N$�O�@pN�2O$sO���N�d*O��N��@NS��O���OmNi��O��O*�ND�AO��N��4No��    �  a  �  �  \  �  j    �  b  ;  �  �  Q  "  1  s  �  +  o  ~  x  �  W  �  �  P  i    1  �  	�  �  �  �  S  E  n  �  d  �    �  #  �  :  �  V  ^  6  �  �  �  �  
�  �  �  �  �  +  R  e  �  �  �      �  �  
�  *  �  �  A  +  C  #  �    �  Q=+<��<�j<T��;D��;�`B;ě�;��
�o�o�o�o�#�
�e`B��o�u��`B�e`B�u��9X��o��C���t����
��1��1��1��1��1��j��`B��/�8Q��h�+��/��`B����`B��h��h��h�o���t��o�t��'C��+�+�+�+�#�
���,1�e`B��P�0 Ž�P�����''#�
�@��'<j�@��<j��C��H�9�H�9�Y��]/�e`B�e`B�u�q�����-��%��hs%����������������X[\hrtu}wtmh[XXXXXX
 !
ABKN[cgrssqjg[NJCBAA>IUXbnnbaUNIF@>>>>>>vz������zyvvvvvvvvvvBHSUamnz���}znaUHC=BCCFO\huv����vuh\OICC����

	�����������?BIN[acec[NMBA>=????"#./149/*#"""""""""��������������������_aemz��������zmha`^_��������������������
#0Un�����{b<0	
�����
��������$//<HUZUTQHE<<;21/$$��������������������������������������������������������������������������������!$$)*)�����������������st|������������}ttss��������������������������������������������������������������������������������EO[hmtwwth[ZOLEEEEEE��������������������������������� 
#+,&#
����t���������������}tt�����������������������������������������������������������������������������������������������������������������������jmpwz���zzomijjjjjj)5>BEEEB54)&&),26<BO[][ZUOJB60)&��������������������#/<HR\ac`K</#������������HNO[gtz����utga[NCHHrt|������������ytsr`ahmz��������zsmea``������������������������������������������������������������W[gnsqg[XSWWWWWWWWWWet��������������thae����������������������������������������*26B[hjrutph[OB62)&*�������������������� #%/;<?AA<</#"    /Ha��������oga\Q<2,/������������������������������������������������*/*$�����X[gt������������g^VX
#$#
/38<HUajoqqlaUH<0+,/COP[fhpttsh[XOLECCCCABN[dgikkga[NLB?:<>A���
#09>@B@<0#
���:<ILQUXYYWUIA<<;::::)5BDOPNKB@5)�����������������������������������������������  �������
"#*.#
�������
#&#

������FKUbn{�������tlUI?AF357BN[cgkkgc[NDB;553}�������������}}}}}}���	�����������������������������������������������@�>�3�0�3�<�@�K�L�S�Q�L�@�@�@�@�@�@�@�@¦¦²²³¿��������¿²¦¦¦¦�!���
�����!�-�/�-�,�"�!�!�!�!�!�!�������������ɼʼҼʼ������������������������������������������Ŀƿſ������������ʼ����������ʼּּ߼���ּʼʼʼʼʼ����������������������������������������ҾA�;�4�3�2�4�5�:�A�M�Q�Z�]�c�d�d�]�Z�M�A�������������������������ʾ˾;Ͼоʾ���àÜàâæìù��������������ùìàààà�A�<�5�0�,�5�A�N�Z�`�g�g�h�g�Z�N�A�A�A�A�<�;�/�&�/�<�H�T�U�U�U�H�<�<�<�<�<�<�<�<�;�8�:�;�H�T�[�T�M�H�;�;�;�;�;�;�;�;�;�;�������������������$�*�)�%�����������������������
���#�%�/�4�<�/�#��
�������o�_�Z�V�O�P�����������������������������������������Ŀѿݿ��ۿѿĿ��������B�6�6�1�2�0�6�>�B�O�V�[�h�h�h�[�O�L�B�B��żŹųŷŹ���������������������ƿ	���������	��"�.�3�;�?�@�;�2�.�"��	��������������������������������������������#�"�������������x�v�l�i�l�x�����������������x�x�x�x�x�x��ƺ�������������$�/�)���������������T�S�H�G�H�N�P�T�Y�a�m�w�r�p�p�m�a�Y�T�T�z�y�t�q�y�z�����������������z�z�z�z�z�zŭŤťŧŭŷŹź��������Źůŭŭŭŭŭŭ�`�`�T�L�S�T�U�`�m�w�y�������y�m�`�`�`�`�����������������������������ĿſпĿ����������������ûлһӻһлŻû�����������Ň��{�x�{�{ŇňŔŠŧŬŨŠŝŔŇŇŇŇ�5�3�,�5�:�B�N�O�N�K�N�S�N�B�5�5�5�5�5�5������������������	���� ����������������������������������������������h�[�O�I�E�M�[�h�tāčĚĢĩĦĚĉā�t�hD�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�D��z�w�n�d�a�]�a�n�v�zÇÇÇ�{�z�z�z�z�z�zÓÏÇ��r�n�i�a�^�a�n�zÓáôüùìàÓƎƁ�u�h�e�h�f�h�uƁƎƚƝƥƧƨƧơƚƎ�O�H�C�9�6�4�6�C�O�\�`�h�h�h�\�P�O�O�O�OìãàÓÎÓàìù��������ùìììììì�����������������������������������������r�m�f�Y�S�M�E�D�M�Y�a�f�r�|�������w�r���������������������������������������侾���������;׾�����	�����	����׾������ԾҾ׾���	���"�.�4�;�<�.�"��������������	��������	����������������������������������������������������������������	������	� ����������������!������������5�1�1�/�5�A�N�S�R�N�A�5�5�5�5�5�5�5�5�5�"��"�%�"��"�/�;�H�K�T�U�T�H�E�;�/�"�"�6�/�3�6�C�O�U�S�O�C�6�6�6�6�6�6�6�6�6�6�{�q�^�U�I�E�A�I�[�b�{ŇŔŞŠŦŠŔŇ�{���~�t�l�q�~�������ֺ�����	����ɺ����v�`�_�d�s�����������ûлܻ�����ﻪ���v�����{�z�{�~�����������������������������b�a�V�V�U�N�V�b�j�o�{�~ǀ�{�p�o�b�b�b�bE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��l�b�t�����ùܹ���4�/������Ϲ������l�ʼü��ʼּ����ּʼʼʼʼʼʼʼʼʼʼ@�4�'������'�4�@�M�Y�\�g�h�_�Y�M�@������Ŀĩč�u�d�X�[�hāĚĦĭ��� ��������
����������������)�1�5�:�A�5��EPEJECE@ECEMEPEQE\E`EdE\EPEPEPEPEPEPEPEP�A�5�(�$��"�)�5�A�N�Z�s���������s�g�N�A�Ľ����������Ľнݽ������ݽнĽĽĽ��)�(�$�*�6�>�B�O�[�b�h�k�i�h�`�[�O�B�6�)�нĽ����������������Ľнݽ�����ݽо���������(�4�A�D�K�A�4�4�(������ùíååêìù������������������������6�6�6�<�<�C�G�O�U�\�b�\�V�O�N�C�6�6�6�6��|�r�r�r�y������������������������ݽн½ŽϽݽ߽�����(�4�8�/�#�������� �����!�-�8�:�:�<�<�:�-���������� �����������������ܻ̻ȻȻ˻ջ���@�N�M�@�4�������/�'�#����#�+�/�<�?�H�P�[�U�S�H�>�<�/�<�:�/�/�/�4�<�H�S�U�U�U�H�>�<�<�<�<�<�<��������������!�.�/�9�2�.�!���ììà×ÓÇÁÇÓàìôùúùîìììì�������������������������������������� @ L = 4   ^ J S F 3 0 S N @ I # C J S  O ? O s W M E � N e / B ; ; / i V K % 4 d C 7 5 8 O J $ % X G D # C I R  : B a N T � V J ) 2 ) + `  X j G < i E  2 A [ 9  H  �  �  x  `  �  N  �  `  �  �  G  6  �  a    :  �  �  d  �  <  u  0    �  �  �  �  �    q    (  �  �  d  8  �  �  �  �  (  �  �  ;  >  F    W  �  �  a  (  �  �  4  �  �  �  &  �  u  c  7  �    W  =      �  �  �  M  �  L  d  e  (  �  }  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�        �  �  �  �  �  �  �  �  �  �  {  o  c  W  K  ?  3  �  �  �  �  �  �  �  |  k  Y  D  ,    �  �  �  K  �  �    \  ^  a  R  A  /      �  �  �  �  �  �  �  �  �    H  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  A  o  �  �  �  �  �  �  p  S  3    �  �  �  R  �  �    �  \  X  T  O  J  B  :  2  (        �  �  �  �  �  y  R  +  �  �  �  �  �  �  �  �  �  �  �  �  �  y  o  d  Z  P  F  <  j  f  _  [  Y  X  X  L  ;  (    �  �  �  �  l  F     �  �  �  �  �  �  �          �  �  �  �  �  r  >    �  �  =  �  �  �  �  �  �  �  �  �  �  �  r  d  W  N  E  =  7  H  Z  b  W  K  ?  4  '      �  �  �  �  �  �  ~  e  M  6      ;  T  m  �  �  �  �  �  �  �  �  x  l  `  R  D  6  +      �  z  f  R  @  /      �  �  �  z  V  /    �  �  z  K    �  �  �  �  �  �  �  �  �  �  e  C    �  �  n  *  �  }  �  2  +  +  J  N  @  *  	  �  �  �  C  �  �  t  .  �  �  �  q    "      �  �  �  �  �  [  '  �  �  W  �  �  2  �  �   �  �  �  �    '  /  1  *    �  �  �  ^  
  �  4  �    g   �  r  s  s  r  q  m  d  S  =    �  �  �  {  K  :  2  "  6  d  �  �  �  �  �  �  �  {  e  O  9  #  
  �  �  �  �  �  �  j  �  �      $  *  +  "      �  �  �  I  	  �  q    |   �  o  W  ?  &    �  �  �  �  �  �  z  Z  :     �   �   �   �   i  ~  p  c  Z  ]  a  f  l  r  �  �  �  �  �          �  �  x  s  n  i  e  a  ^  T  H  <  *    �  �  �  �  _    �  I  �  �  �  �  �  r  M    �  �  �  �  d  ,  �  �  D  �  �  �  W  P  J  C  =  7  1  .  -  +  '           �     G  �  �  �  �  �  �  �  z  p  m  o  p  r  s  u  s  l  e  ^  U  M  E  �  �  �  �  ~  o  `  P  @  /        �  �  �  �  �  �  �  P  I  B  ;  4  -  '  %  '  (  *  +  -  ,  '  "          i  b  \  U  N  G  @  9  1  *  "      	   �   �   �   �   �   �              �  �  �  �  �  �  �  �  g  >    �  �  t       +  0  0  -  '      �  �  �  �  t  A    �    <  �  G  U  c  m  k  i  n  z  �  x  f  T  >  (    �  �  �  �  �  	!  �  �  �  u  	�  	�  	�  	�  	�  	�  	>  �  |  �  `  �      W  V  R  d  z  �    u  g  U  >  "    �  �  �  �  d  =    �  w  �  �  �  �  �  �  p  I  ,    �  �  �  l  	  �  �  �   �  �  �  �  �  �  �  Z  #  �  �  �  N    �  �  b  &  �  �  B  N  R  R  P  K  C  :  1  '        *  L  D  :  -      �  >  C  E  C  <  )    �  �  �  _    �  |  K  )  �  S  �  e  n  j  f  ]  T  J  @  3  #    �  �  �  �  �  i  X  H  7  &  �  �  �  �  �  �  �  �  �    q  a  O  <  &    �  �  �  U  d  d  b  ^  V  K  :  &    �  �  p  8  �  �  �  L    �  �  �  �  �  �  �  �  �  �  �  n  W  ?  (    �  �  �    '  �            �  �  �  v  C    �  �  ;  �  {    �  �  e  �  �  �  �  �  �  �  ~  r  c  S  >  $    �  �  �  V    �        #      �  �  �  �  F     �  d    �  @  �  �  o  �  �  �  �  �  �  �  �  �  c  >    �  �  �  W    �  5  �  �  �  �    1  8  5  &    �  �  �  �  O    �  �  w  G  '      D  W  a  }  �  �  �  �  �  �  l  I  ,    �  �  �  M  H  N  T  U  T  Q  L  G  <  1  %    	  �  �  �  �  �  �  �  ^  Z  V  Q  M  I  E  @  <  8  8  =  B  H  M  R  W  \  a  f  6  <  A  G  I  >  3  (      �  �  �  �  �  �  }  i  T  ?  �  x  q  f  X  O  S  V  A  (    �  �  �  �  j  A    �  Y  �  �  �  �  �  �  �  �  �  �  y  l  ^  K  8  #    �  �  �  �  �  �  �  �  �  �  �  �  m  X  <    �  �  Z  �  t  �  �  �  �  �  �  �  V  "  �  �  a    �  �  F  
  �  *  �  W  e  
�  
�  
�  
l  
3  	�  	�  	O  	  �  �  7  �  K  �  �  &  @  �   �  N  n  �  �  �  �  �  �  �  �  �  h  1  �  �  Z  �  0  �    �  �  �  �  �  �  �  �  t  ]  B  %    �  �  �  �  v  _  H  �  �  �  �  �  �  �  �  ]  -  �  �  =  �    [  �    I  Y  �  �  �  �  ~  b  B    �  �  �  �  �  n    �  >  �  �  �  +  %        	        �  �  �  �  �  �  �  �  �  �  �  R  9    �  �  �  �  y  Y  @  &    �  �  ~  M    �  �  0  _  a  D  '  /  >    �  {  Z  o  �  t  $  �  G  �  �  �   �  �  �  �  �  �  �  �  �  �  �  m  R  9  "  �  �  �  �  ?  �  �  �  �  �  f  ;    �  �  h    �  f    �  P  �  �  '   �  �  �  �  �  �  �  �  �  �  _  4     �    $  �  �  ,  a  �  �  	          �  �  �  �  �  �  �  m  E    �  �  d  /          �  �  �  �  f  /  �  �  q  ,  �  �  8  i  �   �  �  �  �  �  �  �  �  �  p  J    �  �  V    �  X  �     n  �  �  �  �  �  �  �  \  1     �  �  B  �  �  j    �  u    	�  
   
F  
p  
�  
�  
j  
?  
  	�  	�  	g  	  �  G  �  �  �  �  �  *  %          �  �  �  �  �  �  �  �  o  N  )     �   �  �  �  �  �  �  �  �  �  {  q  h  _  W  O  F  >  5  .  )  $  �  �  �  �  �  y  e  Q  <  #    �  �  �  �  �  �  f  3  �  A  &    �  �  �  �  �  p  T  4    �  �  �  �  i  �  �  �  +  #          �  �  �  �  �  �  �  �  �  �  �  t  e  V  C  :  "     �  �  �  �  �  v  j  W  <    �  �  A      �      "  "    �  �  �  �  `  -  �  �  v  #  �  X  �  �   �  �  �  �  �  �  �  �  �  �  m  Y  C  (    �  �  k    �  w  g  o  �  �  �  �  �          �  �  �  ]     �  G  �  �  �  �  �  �  �  �  �  z  Z  9    �  �  �    Z  G  P  I    Q  *    3      �  �  �  �  i  I  &    �  �  �  �  �  