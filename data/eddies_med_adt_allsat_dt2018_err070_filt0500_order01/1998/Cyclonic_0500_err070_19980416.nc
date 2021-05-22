CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�;dZ�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�W�   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��G�   max       =o       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F��\)       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vg\(�       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�F            7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �\)   max       <���       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B1M       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B/��       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C�e       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��#   max       C�        =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          z       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�W�   max       P���       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�|����?   max       ?��;dZ�       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��G�   max       <�h       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @F��\)       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ə����    max       @vg\(�       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @M@           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�            [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_خ   max       ?��;dZ�     0  ^         Z      N         
      
            -                  &   7   z   .      5         
      	   ;         &               *   2      	         #      #                  ;      	      %                           <   '   #   #      N�&*NFi�PW�<NN�P��N�dKO 	�O)CN�N�4�O��.O���N;iP'CNO��mN�j�O�M�N���O���O�`PV��Pa>�P1��O�/WP$�sO�N��Ni�hN�b0N�:P<�RN��zOsYO��O��OۜMN:>�O�-P?vPE&dN��N�M�O��rN�D�P��O��"O�AgOv��NU(�N&��NhcOu&�P��>N��N�\8Oe�O��O�JM�W�N���O�iN�R�Ny��OX|�Nz��OƉ�O��QO��0O�+�N!U�N]��=o<�h<�t�<#�
<o;ě�%   %   ��o�o�ě��ě���`B�49X�49X�D���e`B�u��o��C���t����㼣�
���
���
���
��1��1��1�ě��ě��ě����ͼ�����/��`B��`B��`B��`B��h��h�������+�C��C��\)��P��w�#�
�0 Ž49X�<j�<j�<j�<j�<j�D���Y��aG��e`B�m�h�q���u�u�u�y�#�y�#��%��G������������������������������������������5Nn{zqg[B)����������������������������#Un~������{0
�����	
���������LNR[[gst���}tg[VUSNL4<AHUafntznhaWUH<524#),##!��
 #$#"
�������""#/;HNSMHA</"�������
������mnoz��������znmmmmmm;BWamz���������zaH>;����������������������������������������"/;HINNH;/"	�� ��������������������mo{~�������������{pm)=BN[eglpgNB5������ ��������jny��������������rlj;G\l��������zbUH;/0;�����������7=BNet��������tDB)+7����$)&"�������������#)5>BBB=5)(%########26@BHOOOKEB;60,-2222��������������������!)5Ngz{tqj^[NB@62!��������������������+037<@HUad`a]UH<6/.+�������	 ������������

��������do�������������}tgcdzz����������zzzzzzzz��������������������	*6K\u�������hCFKUn������������nUHF��������������������<BO[cgc\[ODB86<<<<<<�����#241)������ #0<IOUWWUOIG<20##  ��
#0Iclf^UI<
�����������������������BO[hx��������th[OA>B��������������������)-,)%��������������������ABFNO[ed][POIBAAAAAAjpz�����������znjggj)5BO[t��������tN7 STafmtwz~zmfa^TTPOSS��������������������������������������������������������������
#*130*#
������ABOOQOFBA>AAAAAAAAAA����������������������������������������itu�����������}tiiii����������������������'+-*)'����8<HHIOUNH<3288888888���������������������������������������������������).464-����*/<HRNH<3/**********#+/56/'#"�#������#�/�<�H�T�U�H�<�3�/�#�#�#�#�����������ĽνͽĽ����������������������g�O�G�I�Z�g�s�������������������������g��������!�-�3�/�-�!��������������|�m�I�/�(��#�5�s���������(����l�a�f�l�v�y�������������������y�l�l�l�l���������"�(�.�4�A�H�K�A�4�(�����
����
���#�)�/�3�<�@�7�7�/�#��g�b�\�g�t�y�t�g�g�g�g�g�g�g�g�g�g����������������������������������������ā�t�h�e�[�O�6�)�$�%�)�0�6�B�[�h�t�{�zā��������������������������*�-�)����U�L�H�B�?�<�1�<�C�H�U�W�W�U�U�U�U�U�U�U����ƧƖƌƋƖƛƓƚƧ����������������������������$�0�3�;�9�9�2�3�0�$����������������������������������������������������������������������������׼ּʼ¼��������ȼüʼӼּ׼޼�����ֿ������y�m�Z�R�T�X�`�����������ÿ��������Ŀ������������������Ŀѿڿ���߿ݿѿļ��r�p�{�����ּ����-�7�:�6�.���ּ������������þõù����)�B�O�[�a�d�\�6���2������ݿ����ѿ���5�L�Z�g�z�u�N�A�2��׾ʾ����ʾ׾��	���%�-�/�.�"�����׾��������ƾ;׾��	�"�&�-�,���	���׻�������ֺպѺֺ����������àÚÓÒÑÓÔßàì÷ùûüùöðìàà��úý�����������������������������������/�(�"��"�/�/�/�;�H�K�T�Z�T�H�;�/�/�/�/����ŹŮŹ����������������������������������������������6�\ƁƔƗƒƁ�u�h�C�*�ìààÛÙÛàìðù����������ùìììì����������������������������!�����D�D�D�D�D�D�D�D�D�D�EE
EEED�D�D�D�D��I�F�I�T�V�b�o�v�{ǁǈǔǓǈ�{�v�o�b�V�I�m�a�T�L�M�R�g�m�z������������������z�m�;�;�/�"� �"�/�/�;�H�J�J�H�H�;�;�;�;�;�;��������(�4�<�A�G�D�A�;�4�(����"�	��������,�2�1�;�G�T�^�[�f�R�;�.�"�������������������3�?�6�6�'���ѹ��'�����'�3�?�@�E�@�3�'�'�'�'�'�'�'�'�����������������������������������������Ľ��������������������н�����������ľ�������������$�(�(�(��������F������!�-�S�l�}�����j�^�W�X�]�S�F�"�#�.�5�;�H�a�m�z���������z�a�Z�T�H�/�"�l�`�`�g�l�s�y�{���������������������x�l�ݿۿ׿׿ݿ������������� �����¿¼³¶¿��������������¿¿¿¿¿¿¿¿��������������������������������������ٹܹڹϹù¹��ùϹܹ�����ܹܹܹܹܹ��s�g�X�P�Z�g�}�������������������������s����������µ¨²������/�U�`�R�N�<����������������
���!��#�&�#���
�������0�/�)�-�0�<�=�I�U�a�]�U�S�I�D�<�0�0�0�0���������ɺֺ������������ֺɺ��Y�@�5�'�"�$�2�6�7�@�M�W�a�f�i�|������Y��ܻлŻ����ûлܻ������'�0�'���������"�&�.�2�.�"�����������t�k�h�`�e�h�tāăčččĊā�t�t�t�t�t�t�3�0�3�5�?�@�M�Y�e�r�t�t�r�i�e�Z�Y�L�@�3�����������������%��������Z�P�N�D�N�Z�g�p�s�������s�g�Z�Z�Z�Z�Z�Z�Y�M�@�>�=�?�D�M�Y�f�r�y�������t�r�f�YD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D���ܻλƻûͻ�����4�>�<�2�'�
�� ����R�F�D�C�G�S�_�l�x���������������x�l�_�R�G�:�!����.�G�S�`�l���������y�n�d�S�G����ݽĽ��������нݽ��(�;�=�4�(�����ʾɾž��ʾ׾ھھ׾Ͼʾʾʾʾʾʾʾʾʾ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� I $ 4 J l 1 A < / 0 " Z r P = @ : l : 7 E < � T C I P , N m T . G 7 T 6 J  u ; > 2 Z ` O Z a I A n L + > 7 V i I Q F , [ I h - = F 6 H P n :    �  [  �  k  	�  �  5  q  9  �      �  H  H  �  |  L  �  �  �  �    u    ?  /  p  �  
  �  �  �  `  :    ]    �  �  �  �  3    �  r  �     m  _  �  �    �    &  �  k   �  �  D  �  �  �  y  �  5  �    {  }<�1<��ͽ�t�<o��\)��o���
�t���`B�D���C��ě��e`B�aG���w��j�o�C��<j�]/��t��\)��+�,1����L�ͽ�w����`B�+��t��8Q콁%�8Q�P�`����w��O߽����\)��w�T���,1��C���o��O߽aG��49X�,1�T����o���ixսaG���\)���罓t��P�`��C���C��y�#��o�� Ž�����ȴ9�\�\��7L����BP�B*�.B YB,F�B%�3B.5B	5]B~�B��B$Y�B9`BbMBl1A���BvB�^A��B��B*KTB�|B,�VB��A�z=B��B	��B3-BtB[B�B�$B�B�hBA�B��Bb&BajB�B �-B1MB�B eB��B#��B&A�B%t�B�UB=�B}BP�B�FB�wB�4B	q�A��cB�B1�B �ZB$V_BmBC�B �]B
hB�BB[B�6B3[B�NBe�BňBϑB>�B*��B�=B,X�B&E�B.8�B	@�B�_B��B$?B�HBy�B�/A��:B�B�BA���BU�B*��B@�B-"�B��A��BB	��B@�B>B<xB��B?�B�B��BAvBYmB?�BrB�B ��B/��B:1B��BέB#zyB&�BB&S�B��B��B~B@6B'}B=�B��B	ȨA�w�B:iB��B �/B#�<BA�B@YB 9�B
�FB��B�B�BÌB?+B=B{�B0zB��A�sqA%k6A�G�@i$�A��A�oA7FjA�]�A�H@��A�L7A��A�=EBA�B	sA�vA�� A j�Ao�Ax;6A?mA��A��UAX�AX��@OA�A���A�M�A��;A�%BA��?A�đA���C�5�B�A�s*A�H"A6�iAcNE>��?���A��"A&z�A2�@�ۋA��+@�*A�v�A�6�Bq�>��A��KA��A�p0A���@H�,@�ɲ@��VA^�tAܿE?��A�XMA�r�@�ǁC��U@���@�S~A��A0F>AQ�C�eA�B�A%${A�^h@kќA�KAA�A8�A�VA�~�@���A�`A��A�~�BtjB	S�A�A�x�A ��AnsAvR�A�TA�'�A��AW�SAX��@LAˏ�A�wA���A��B "fA���A�=C�7gB�"A���A�`�A7 �A]!>��?�}�A�"MA'�A3.@{o�A���@�	LA�WA���B��>��#A��4A�zA���A�@C��@���@�:�A^��A܂d?���A��yA���@�-C���@��@� ;A�A1�\AQ*�C�          [      O         
                  -                  &   7   z   /      6         
      
   ;         &               *   3      	         $      $                  <      
      &                           =   '   #   #               +      Q                     #      /               #   !   5   3   5      )                  /               %         /   /         )      )                        ;            !                           %      !   %               !      E                     #                        !   5                                             #         /   +         '      )                        +                                                      N��NFi�O��NN�P���N���O 	�N�iN�N�VO[�(O�#N;iO��TO��[N�j�O��N�?O���O�`PV��OwOt
�O![`O9�N�>pN�c�Ni�hN�b0N�:O�;�N��zOa�`O��N�6POÓ4N:>�O�-P?vP#8�N��N�M�O��hN�D�P��O{��O�AgO`%'NU(�N&��NhcOQ�P&'xN��N�\8Oe�OP�\O}.�M�W�N���O�iN�R�Ny��OC��Nz��O2ǕOt�O��[O��N!U�N]��  b  �  �  B  c  �  �  �  6    	  �  �  �    �  Z  �  �  8  �  �  f  �  %  �  �  ?  �  �  �  �  �  	�  �  �  �  �  �  �  �  �  �  d  �  �  �    �  �  ~    Z  ;  ,  Q  �  ?    �  g  e  ,  P  �  �  u  �  �  x  �<�h<�h�T��<#�
�o;��
%   �D����o�D���49X��`B��`B��`B�T���D����o��o�ě���C���t������w��`B�T����9X�ě���1��1�ě��H�9�ě�����������`B����`B��`B��`B�C���h���+���+��P�C��t���P��w�#�
�8Q�u�<j�<j�<j�aG��H�9�D���Y��aG��e`B�m�h�y�#�u���罃o��%��O߽�%��G������������������������������������������5BNbfge[B)�������������������������#n{�����{n<#���������������LNR[[gst���}tg[VUSNL9<HUXalcaUH><8999999#),##!��
"
���������#/<IOLHD</)"�������
��������mnoz��������znmmmmmmRX\amz������zmaTJLNR����������������������������������������"/;?HIKHG;/"	������������������������������������|}��)=BN[eglpgNB5������ ����������������������������Zamz���������zmaYTUZ������
����Z[ghtz������ytqg][XZ����#'%!������� �������#)5>BBB=5)(%########26@BHOOOKEB;60,-2222��������������������()5BNU[__][WNB?5.(&(��������������������+147<BHU_c``\UH<80.+�������	 ������������


��������it���������������tgizz����������zzzzzzzz��������������������	*6K\u�������hCHUn������������}nUIH��������������������<BO[cgc\[ODB86<<<<<<�����#.1/
������� #0<IOUWWUOIG<20##  ��
#0Iclf^UI<
�������������������������BO[hx��������th[OA>B��������������������)-,)%��������������������ABFNO[ed][POIBAAAAAAknrz����������znlihk39Uait���������gNB93STafmtwz~zmfa^TTPOSS��������������������������������������������������������������
#&-.)#
�������ABOOQOFBA>AAAAAAAAAA����������������������������������������itu�����������}tiiii����������������������&*+)(#���8<HHIOUNH<3288888888����������������������������������������������������
)+/0.+&���*/<HRNH<3/**********#+/56/'#"�����#�/�<�F�H�J�H�<�/�#�����������������ĽνͽĽ������������������������s�g�_�Y�Y�_�s��������������������������������!�-�3�/�-�!��������������y�N�I�K�Z�s���������	��	���	��l�e�i�l�y�y�������������������y�l�l�l�l���������"�(�.�4�A�H�K�A�4�(�����
����#�/�0�9�3�/�-�#�������g�b�\�g�t�y�t�g�g�g�g�g�g�g�g�g�g�����������������������������������������6�/�)�'�(�-�6�B�O�[�h�w�w�t�h�`�[�O�B�6������������������������"�)�,�&����U�L�H�B�?�<�1�<�C�H�U�W�W�U�U�U�U�U�U�U����ƳƦƛƚƧƮƳ������������������������
�����������$�0�1�:�8�8�1�2�0�$������������������������������������������������������������������������������׼ּѼʼ¼��������ʼּݼ�����ּּּֿy�a�X�[�`�m�y�������������������������y�Ŀ������������������Ŀѿڿ���߿ݿѿļ��r�p�{�����ּ����-�7�:�6�.���ּ���������������������)�6�<�H�D�B�6�)�����������������(�5�B�G�G�A�5�(��	�����׾Ͼʾʾ׾����	�
���#�"��	��������������	������	������������ֺֺҺֺ�����	�����àÞÔÓÒÓ×àìøùúùóìéàààà��úý�����������������������������������/�(�"��"�/�/�/�;�H�K�T�Z�T�H�;�/�/�/�/����ŹŮŹ��������������������������������� ��������6�O�T�\�d�`�\�Q�C�6�*�ìààÛÙÛàìðù����������ùìììì���������������������������������D�D�D�D�D�D�D�D�D�D�EE
EEED�D�D�D�D��V�J�J�U�V�b�o�y�{�~ǈǓǒǈ�{�t�o�b�V�V�m�a�W�T�N�O�U�m�z�������������������z�m�;�;�/�"� �"�/�/�;�H�J�J�H�H�;�;�;�;�;�;��������(�4�<�A�G�D�A�;�4�(����"�	��������,�2�1�;�G�T�^�[�f�R�;�.�"���������������ܹ�����-�&����Ϲ��'�����'�3�?�@�E�@�3�'�'�'�'�'�'�'�'�����������������������������������������Ľ������������������н���������ľ�������������$�(�(�(��������F������!�-�S�l�}�����j�^�W�X�]�S�F�/�,�(�,�1�8�;�H�T�a�m�{�����z�a�T�H�;�/�l�`�`�g�l�s�y�{���������������������x�l�ݿܿ׿׿ܿݿ������������������¿¼³¶¿��������������¿¿¿¿¿¿¿¿��������������������������������������ٹܹڹϹù¹��ùϹܹ�����ܹܹܹܹܹ��s�m�g�Z�S�Z�g�s�����������������������s�#�
����¿¶³º¿�����
��I�G�C�A�;�/�#�������������
���!��#�&�#���
�������0�/�)�-�0�<�=�I�U�a�]�U�S�I�D�<�0�0�0�0���������ɺֺ������������ֺɺ��G�@�4�/�-�4�<�@�M�Y�b�f�r�v�}�z�r�f�Y�G��ܻлȻ����ûлܻ�����"�'�����������"�&�.�2�.�"�����������t�k�h�`�e�h�tāăčččĊā�t�t�t�t�t�t�3�0�3�5�?�@�M�Y�e�r�t�t�r�i�e�Z�Y�L�@�3�����������������%��������Z�P�N�D�N�Z�g�p�s�������s�g�Z�Z�Z�Z�Z�Z�Y�M�@�@�>�@�E�M�Y�f�o�r�������~�r�f�YD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����޻������'�)�/�'�%��������_�S�H�F�K�S�_�l�x�����������������x�l�_�G�:�!����.�G�S�`�l�y���y�v�l�b�`�S�G�����ݽн��Žؽݽ�����(�1�6�.�(���ʾɾž��ʾ׾ھھ׾Ͼʾʾʾʾʾʾʾʾʾ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� P $ 0 J d / A C / ) " Y r 7 < @ : Y : 7 E T S T ' E F , N m @ . E 7 R 1 J  u 0 > 2 _ ` O S a G A n L + > 7 V i 3 M F , [ I h * = C 0 > A n :    �  [    k  	  �  5  �  9  �  �  �  �  W  #  �  6  �  $  �  �  !    �      �  p  �  
  4  �  �  `    �  ]    �  �  �  �  �    �  ,  �  �  m  _  �  �    �    &  �     �  �  D  �  �  �  y  ~  �  P  !  {  }  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  6  E  R  Z  _  a  `  ]  Y  V  U  P  J  @  6  -  #        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  i  ^  R    �    q  �  �  �  �  �  �  _    �  e  �  �  �  1    �  B  A  @  ?  >  =  <  ;  :  9  5  -  &              �   �  �  �  4  R  b  D  (  �  �  o  )  �  d  �  ]  9    �     ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  5    �  �  �  �  �  �  �  �  �  �  �  }  v  l  b  X  B  *     �  �  �  �  �  �  �  �  �  �  �  �  p  b  V  O  H  ?  4       6  $    �  �  �  �  �  �  l  Q  6    �  �  �  �  �  c  C                  �  �  �  �  �  �  f  M  6      �  �  �      	     �  �  �  [    �  t  '  �  �  I  �  /  v  �  �  �  �  �  �  �  �  n  O  1      �  �  �  �  �  f  =  �  �  �  �  �  �  �  �  �  �  �  y  e  P  ;  (         *  D  a  n  v  �  �  �  �  �  �  �  �  e  /  �  {  �  o    �          �  �  �  �  �  \  .  �  �  `    �  2  �  ?  �  �  �  �  �  s  _  S  I  3  "      �  �  �  �  �  }  f  O  S  X  Z  W  Q  G  <  -      �  �  �  �  {  ?  �  �     l  �  �  �  y  k  X  J  )    �  �  u  C    �  �  �  Q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  B    �  �  e    �  8  7  1  !    �  �  �  �  �  y  a  C    �  �  4    �  �  �  �  �  �  �  _  0    �  �  �  �  [  !  �  t  �  �  �  Y  
�  �  @  �  �    �  A  |  �  w  2  �  r  &  �  �  
�  	K  _  ]  U  u  �  �  8  T  c  e  X  L  >     �  {  	  �    I  v    M  v  �  �  �  �  �  �  �  �  �  n  7  �  �  Z    �  \  u  �    4  U  n  ~  �  �  �        �  �  ^  �  X  <  �  �  �  �  �  �  y  _  B  :    �  z  1  �  �  H  �  �    �  }  �  �  �  �  �  j  M  .    �  �  �  W    �  z    �  �  ?  5  +      �  �  �  �  �  g  8    �  �  h  /  �  �  }  �  �  �  �  t  e  U  C  0       �  �  �  �  |  Z  9     �  �  �  �  �  q  f  ]  U  L  D  =  9  9  6  .  %      �  �  G  �  �  #  B  Y  u  �  �  �  �  X  !  �  �  N  �  #  3  �  �  �  �  �  �  �  �  q  \  F  .    �  �  �  �  �  �  w  c  �  �  �  �  �  n  V  <      �  �  �  q  >    �  R  �  �  	�  	�  	�  	x  	F  	  �  �  O    �  y  +  �  �  &  �  O  �  �  �  �  |  o  W  3    �  �  �  O    �  �  D  �  �  2  �  `  �  �  �  �  �  y  b  F  (    �  �  s  *  �  k  F  &      �  �  }  w  q  l  f  `  [  U  P  J  E  ?  :  5  /  *  $    �  �  �  �  �  �  �  �  �  �  o  X  A  (    �  �  �  �  @  �  �  d  H  '    �  �  �  �  �  �  _  .  �  �  F  �  #  !  �  �  �  �  �  �  b  *  �  �    M    �  �    �  	  y  b  �  �  �  �  �  �  x  k  ]  P  D  9  .  $               �  �  �  �  �  �  �  �  �  �  {  h  T  A  .      �  �  x  �  �  �  �  �  �  �  y  a  G  +  
  �  �  �  �  L    �    d  a  ^  _  W  G  4      �  �  �  �  _  8    �  �  s  :  �  �  �  �  �  �  �  �  �  y  Y  1  �  �  n  B    �  i  �  �  �  �  �  �  ~  r  d  ]  R  <    �  �  �  n  �  S  �  �  �  �  �  p  M  )    �  �  �    O    �  M  �  �  }     o  �      �  �  �  �  �  �  �  i  F  "  �  �  �  u  E    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  R  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  w  k  ^  Q  F  :  ,    �  �  �  w  I    �  �       �  �  �  �  �  �  s  @    �  �  g  -  �  �  y  �  G  :  3  =  S  Z  Q  ?    �  �  L  �  �  �  �  	  m  �  K  ;  "  	  �  �  �  �  �  �  �  l  W  D  +    �  �  �  U  $  ,  '  !    �  �  �  �  �  �  s  X  <    �  �  m    d   �  Q  H  /    �  �  u  >    /  �  �  �  �  Y    �  �  �  R  �  �  �  �  �  �  �  �  �  �  �  \  #  �  �  Z    �    F  $  8  ?  :  1  "    �  �  �  v  H    �  �  T  �  �  �  �    {  w  s  o  k  g  b  [  T  N  G  @  9  0  '          �  �  �  �  �    d  A    �    "  �  N  �  S  �  C  �  &  g  T  A  /      �  �  �  �  �    e  J  0      �    �  e  `  [  V  P  G  >  4  *        �  �  �  �  �  �  �  �  ,  '  "      	  �  �  �  �  �  �  �  �  �  �    �  �  �  E  O  J  7    �  �  �  w  D    �  �  n  &  �  �  7  �  �  �  �  �  �  �  �  �  s  b  R  @  -    �  �  G  �  �  .  �  l  �  �  �  �  �  �  �  �  �  �  w  2  �  Z  �  7  x  ;  �  Q  l  u  h  U  ?  +    �  �  �  _  *  �  �  G  �  /  �  �  l  �  �  �  {  Z  4    �  }  >    �  �  j    �  `  �  �  �  �  �  �  �  �  �  �  �  �  �  s  J  !  �  �  �  E  �  �  x  s  o  j  f  a  [  V  P  K  B  6  *        �  �  �  �  �  �  �  {  j  W  B  +    �  �  �  �  �  k  I  %  �  �  �