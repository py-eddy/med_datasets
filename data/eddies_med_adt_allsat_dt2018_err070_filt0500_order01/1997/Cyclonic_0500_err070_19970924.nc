CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��G�z�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�z4   max       P�h       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <D��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>xQ��   max       @F�          
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ٙ����    max       @vUG�z�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q            �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �!��   max       ;��
       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�O�   max       B4�N       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��8   max       B4�       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >]Ы   max       C�m�       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��1   max       C���       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�z4   max       P�7�       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*�0�   max       ?ӕ�$�/       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\   max       <#�
       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>xQ��   max       @F���
=q     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ٙ����    max       @vUG�z�     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q            �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�PH   max       ?ӓݗ�+k     �  \8   
                           G                     `         0               !   	         i      /      
      -                  !            	   ,   
   
         
            	   g   /      ]                        )      "N�3(O�m:Nf)�N:�O)	�N?�@O�[�N[��N�XP'`OOBOʟN��OUN�UNz�P�hNb��O(P�7�NtŻON��N-��M�z4O�N�3KPiO6hP x�Op�P9�N.%�N���O"D)Pe�O��fO��N<D�O�2�O	)vO^��O�NG��N�5Nذ$O��FOn�O<<N��N��N�wOv�Nե�N���O>RP9|P^�Oc��P[^�O)�O��N��gN�+�O� OqH5No�O��N�lbOF.<D��<#�
<t���o�#�
�49X�49X�D���T���e`B�e`B�u�u�u��o��o��C����
��1��1��j��j�ě��ě��ě��ě���/��/��`B��`B��`B��h���o�+�+�+�+�+�C���P��w��w�#�
�'0 Ž49X�H�9�H�9�L�ͽP�`�P�`�Y��Y��Y��e`B�e`B�u�}�}�}󶽁%��O߽�t����P���㽧�-������������������������������������������������������������`agjmxvqmda^````````#/<CHIHDB@</$$#05BNQPNB5.0000000000�����������������

��������������������������������
#/<DLQYU</
�����������������������������������������)5:BGHB5))#)5=BNN[[`[XTNB52)#)*24) ttu{������utttttttttnbIB2#�0Ud{�����tn66)#%)66666fhnt����������tkhdff��0U{�����{WI0��T[hjt����th[[XTTTTTT#*6CDGGFC@6*

���		���������������������������������������������������������������������EHTmz����������maUOEtz{�����������zuprtt������"!�����������%('&
����)Bht|~������h\VNB))#+09<0# ��������������������������

��������@[t����������tg[GAC@R[ag�����������tgZRR��������������������ghqt����thfcgggggggg��������������������MQ[^gtv}}vtsge[YVRNMx��������������zuprx�")67>;6/)��)/6796)&Y[_gtx}{vtrnlgeda^[Y2;HRTVTRLH?;;50/2222���������������wz���������������zrw�����������������������������������������������������������yz�����������zptyyyy������������������������������������������������������������nu{����������{qunmmn������������������,45,�������#)/242,#
����v��������������tmlnv������������������������ ���������������������������<ACCBA</-//,)&#""#/<��������������������5<Hanz�������qaUH<75��������������������)5BOVY\YWNID5)%$)��������������������	
#/7;530+#
	�����������������Ⱦʾξʾʾ�����������������������¿�����������#�4�<�?�/�#����0�,�*�-�0�7�=�D�F�D�=�;�0�0�0�0�0�0�0�0�B�?�6�)��)�6�B�G�O�S�O�B�B�B�B�B�B�B�B�6�5�,�+�0�6�I�O�Q�[�h�t�z�y�t�h�[�O�B�6�A�;�:�>�A�N�R�V�T�N�A�A�A�A�A�A�A�A�A�A���r�z���������������ĿȿʿƿƿοĿ������=�8�=�A�I�V�X�b�f�b�V�I�=�=�=�=�=�=�=�=�M�C�A�6�=�A�M�U�Z�`�f�j�f�Z�M�M�M�M�M�M�s�i�g�Z�\�e�i�r������������� ���������s��� ����ݽʽ������Ľн����	��� ��s�l�g�[�Z�W�U�Y�Z�g�s�}���������������s�`�`�`�g�j�m�r�y�����������{�y�m�`�`�`�`�������������������������������������������������������������������������������m�e�`�T�S�T�`�m�y�~�y�q�m�m�m�m�m�m�m�m�N�X�N�A��罷���{�n�z���������н��4�N�L�L�Q�Y�[�Y�T�L�@�3�2�3�8�@�H�L�L�L�L�L�<�9�/�*�#�"� �#�/�<�H�U�V�^�^�U�O�H�<�<�����������a�D�1�0�7�Z����������������𻅻}������������������������������������������ܾ�����	���#�*�,�+�"��	��������������	�����	���������������������������������������������������������@�4�&�#�+�>�M�f�r�u��������������r�Y�@�G�=�;�4�:�;�G�R�T�Y�`�m�o�n�m�`�Y�T�G�G������������� �	��;�H�W�^�\�R�S�X�H�"��Ç�{�z�v�v�zÇÇÓØàìñïðìàÓÇÇ�:�/�:�F�S�]�h�c�l�������������������S�:�-�-�!���	���!�-�:�F�R�M�F�@�:�/�-�-�|�u�u�~���������	������	���������|�û��ûȻлܻ��ܻлûûûûûûûûû�ƧƦƦƧƳ������������ƳƧƧƧƧƧƧƧƧ�_�]�S�F�;�@�S�_�l�x�}���������~�x�t�l�_�;���� �0�H�T�m�z�}�w�v�������m�T�H�;���������������	���!�)�,�*���	���ʾ¾������������ʾ׾���������׾ʾʺ�	�����'�*�-�)�'����������3�+�/�9�>�A�L�Y�e�~�����������~�r�Y�@�3����������������&�)�5�5�8�5�)�������������������������$�%�$��������T�S�J�H�>�;�8�:�;�H�T�a�m�u�y�q�m�d�a�Tìçéìöù��������ýùìììììììì�O�H�C�<�?�C�O�\�h�q�uƁƎƘƎƁ�u�h�]�O�5�)�*�5�:�A�G�N�Z�c�g�h�g�Z�N�A�5�5�5�5�B�8�O�W�tāĚĦĳĿ����ĿĸĦč�t�h�Z�B�������������������
��"� ����
���������������*�6�C�O�C�C�6�+�*����ɺɺ����������ɺֺٺ������ߺֺɺ��������������!�)�*�1�*��������������������������������������������������H�@�;�8�8�:�>�C�H�T�a�l�u�x�y�z���z�a�HŭŬŠŔŉŔŠŪŭŹ��������������Źŭŭ�=�<�0�$��������$�(�0�3�=�>�>�=�=�����������������ûлܻ����ܻӻлû��Y�M�E�A�A�M�Y�r�����ս����ּ����r�Y�`�=�0�1�G�f�y�����н�������Ľ����`ED�D�D�D�D�D�EEEE*E7ECEEEAE7E/E*EE�g�]�]�c�q¦¿������������¿�e�`�l�~����������ü���������������r�e�ּ̼ռּ������������������ìäàÝÝàãìðù����������ùììììD|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DyDtD|��	���(�5�7�<�:�9�5�4�(����������������������ùܹ�� ����ֹܹй¹������'�� �'�3�@�A�@�=�3�'�'�'�'�'�'�'�'�'�'�^�W�S�U�aÇÓàìðù������ùìÓ�z�n�^���
��������
��#�0�0�<�?�@�<�0�#���ݿܿѿϿпѿݿ�������!�������� 2 l m � B B S = N H o 4 l ? I P I m # ` c 1 d J , ' H  ? * q P  2 1 I R A b 1 H @ l u > � [ T * 9 P B g ] 1 W Y , B j \ + x = | 8 E V 9    �  �  �  T  x  g  �  w  �      U  �  >  4  5  �  �    V  �  �  Z  
  Z  �  �  /  �  0  �  W  �  d  q  7  ;  a  �  6  �  ^  �  I  �    \  6  �  �  �  �      �  �    �  �  �  \  �  A  g  �  "  �    �;D���ě�;��
�ě���`B�e`B����t���C���1����󶼴9X��9X��1��1��`B��/�8Q콋C���`B�49X��/��`B�ixս+�ixս#�
�+�@������+�'49X���w�y�#�H�9�0 ŽY��H�9��\)�T���<j�P�`�L�ͽ� Ž]/�q���y�#�m�h�y�#����u�q���y�#�!�����`��-��-���w����w�ě��Ƨ�-���
���m����oB4�NB�Be�A�M�B;�B��B,��B��B=iB��B!�XB�_B�nB�AB�B��B&�fB��B�B&��B�3B/�'A�O�B_�B s�B!l�B >.B R�B�6B[B��B%�B��B#N�B	�rB
-BTaB/)B ��B	1QB �BЌBB	\�A��]BmB �kB��B#�B��B�LB��B�B7�B)0nB�]B�NBEsBl$B+�DB.7�B�XBHvB��B+)B��BtGB9�Bz�B4�B@BOiA�[B�BB�_B-�B�*BB B�3B!��B�?B�iB>�B7`B��B'A>B��B��B'�mB��B/�A��8BDaB ��B!��B?,B {�B��BA�B?�B%M�B�LB#@GB	<�B
@�B�B88B!�B	>B CB�BŪB	C�A���B�\B ��BS�B#?�B�TBíBf3BBA]B)L�B��BEOB��B�`B+�|B.�B�SB?�B��B�OB~hB8B/�B?�AMH�A��
B
~*A��BA٥kA��As6gBj�A<��A�e�A+�A�;�Al1AI��AI�Aj<�A*?��A���A�r�@�U7AZ�A�J�Aq.j@۪PAf��A�T@Aʑ@��@s�}A��v@�WB��@��A�Q�AZ�gAR-�?��i?��A��
B.]A��JA�_�B��A��A�m�A�.A�@;
A��RA�DkA��"A�X�B	v�@��}@�k�A!�tC�m�A��@雙A��A�wC�ՌA�
>]Ы?��A��`A��A�w�ALWBA�|sB
��A�׍Aق�A�wAt�PB��A;�7A��A,��A��;Am�AGC�AI�&Ak*A+��?�qcA�o�A��F@�<�AZ�yA��Aq �@�8lAf�A��A�N�@�"�@s��A�]�@���B��@� A�̯A[�AQI�?��I?�A�~BP�A���A̒�BRA���A� �A�X`A�x1@4�A�w�A�y�A�(A�s�B	>\@�J@�4A'�}C���A��A@�8A�IA��C���A��;=��1?��$A�9#A��A�y-   
                           G                     `         0               "   
         i      0            .                  "            
   ,            	               	   g   /      ]                        *      #                     !         +                     @         C                     '      )      3            +            #                     #                              -   7      /                  !                                                               /         C                     %                        %            #                                                   +   7      )                  !            N5�O�m:Nf)�N:�O)	�N?�@O��mN[��N2nFO{��O4'OʟN��N�{N�UNz�P��Nb��Nè�P�7�NtŻON��N-��M�z4N�(CN�3KP'.O6hOF��N���O]cN.%�N���O"D)P��O8�O��N�O�2�N�/ O7ZzO�NG��N���Nذ$O��
On�O<<N��N��N�wOv�Nե�N��%O>RP�CP^�Oc��P,O�EO��N�P�N�+�O� OqH5No�O��N�lbO:�M  G    �  I  �  �  ?  &    �  o  �  �  #  f    �  �  �  �  �  �  �  t  �  �    �  �  L  .  a  A  3  �  k  9  �  ~    �  p  H  �  �  	S  B  �  p  �    �  �  �  %  �  1  z  	�  v  .  �  	$  �  �  �  �  �  �<o<#�
<t���o�#�
�49X�D���D���e`B�8Q켋C��u�u��o��o��o�L�ͼ��
���ͼ�1��j��j�ě��ě��#�
�ě���h��/��t����@���h���o�\)��w�+�C��+�t��'�w��w�,1�'@��49X�H�9�H�9�L�ͽP�`�P�`�Y��]/�Y���C��e`B�u���T��%�}󶽅���O߽�t����P���㽧�-�\���������������������������������������������������������`agjmxvqmda^````````#/<CHIHDB@</$$#05BNQPNB5.0000000000������ �����������

������������������������������
#/5;>>><7/#
��������������������������������������)5:BGHB5))+5@BNY[^[VQNB55-++++)*24) ttu{������uttttttttt0<IYcwzwumbUI:/#66)#%)66666ghiqt��������tnhgggg��0U{�����{WI0��T[hjt����th[[XTTTTTT#*6CDGGFC@6*

���		���������������������������������������������������������������������MTmz�����������maXRMtz{�����������zuprtt�����	�����������#&%#�����46BO[hlrsqnh[OB=6314#+09<0# ��������������������������

��������A[t����������tg[HBDAY[gt����������tqgc[Y��������������������cht����thgcccccccccc��������������������S[bgstz{tsng[[UPSSSStz{�������������zwrt�")67>;6/)��)/6796)&Z[bgtu{yttnhggc`][ZZ2;HRTVTRLH?;;50/2222��������������wz���������������zrw�����������������������������������������������������������yz�����������zptyyyy������������������������������������������������������������nu{����������{qunmmn������������������,45,�������#)/242,#
����sx|��������������wts������������������������ ���������������������������<ACCBA</-//,)&#""#/<��������������������5<Hanz�������qaUH<75��������������������)5BOVY\YWNID5)%$)��������������������
#,/952/*#

�����������������þþ�����������������������������¿�����������#�4�<�?�/�#����0�,�*�-�0�7�=�D�F�D�=�;�0�0�0�0�0�0�0�0�B�?�6�)��)�6�B�G�O�S�O�B�B�B�B�B�B�B�B�6�5�,�+�0�6�I�O�Q�[�h�t�z�y�t�h�[�O�B�6�A�;�:�>�A�N�R�V�T�N�A�A�A�A�A�A�A�A�A�A���x�}���������������ÿǿɿſ������������=�8�=�A�I�V�X�b�f�b�V�I�=�=�=�=�=�=�=�=�M�G�A�8�?�A�M�R�Z�]�Z�X�M�M�M�M�M�M�M�M�����x�w�{�������������������������������нϽĽ����ĽȽнݽ޽�����������ݽ��s�l�g�[�Z�W�U�Y�Z�g�s�}���������������s�`�`�`�g�j�m�r�y�����������{�y�m�`�`�`�`���������������������������������������������������������������������������������m�e�`�T�S�T�`�m�y�~�y�q�m�m�m�m�m�m�m�m���������������Ľн����5�:�-��ݽĽ����L�L�Q�Y�[�Y�T�L�@�3�2�3�8�@�H�L�L�L�L�L�H�E�<�/�-�&�,�/�<�H�R�U�\�[�U�I�H�H�H�H�����������a�D�1�0�7�Z����������������𻅻}������������������������������������������ܾ�����	���#�*�,�+�"��	��������������	�����	���������������������������������������������������������Y�O�M�F�@�?�@�M�Q�Y�f�j�r�v���r�o�f�Y�G�=�;�4�:�;�G�R�T�Y�`�m�o�n�m�`�Y�T�G�G�	������� ���!�;�H�U�\�Z�T�Q�P�U�H�/�	Ç�{�z�v�v�zÇÇÓØàìñïðìàÓÇÇ�x�s�l�o�x�����������������������������x�!������!�-�:�F�N�H�F�;�:�-�!�!�!�!�����������������������������������������û��ûȻлܻ��ܻлûûûûûûûûû�ƧƦƦƧƳ������������ƳƧƧƧƧƧƧƧƧ�_�]�S�F�;�@�S�_�l�x�}���������~�x�t�l�_�;����#�/�;�H�T�m�y�u�t�������m�T�H�;�	�����������	���"�$�(�$�"����	�ʾ¾������������ʾ׾���������׾ʾʺ�����'�'�,�(�'�����������3�+�/�9�>�A�L�Y�e�~�����������~�r�Y�@�3����������������)�0�)�)��������������������������������"�"������T�S�J�H�>�;�8�:�;�H�T�a�m�u�y�q�m�d�a�Tìçéìöù��������ýùìììììììì�O�M�C�?�B�C�O�\�h�h�uƁƄƁ�u�h�\�T�O�O�5�)�*�5�:�A�G�N�Z�c�g�h�g�Z�N�A�5�5�5�5�O�O�[�tāĉĚĦĳĿ������ĿĳĦč�t�[�O�������������������
��"� ����
���������������*�6�C�O�C�C�6�+�*����ɺɺ����������ɺֺٺ������ߺֺɺ��������������!�)�*�1�*��������������������������������������������������H�@�;�8�8�:�>�C�H�T�a�l�u�x�y�z���z�a�HŭŬŠŔŉŔŠŪŭŹ��������������Źŭŭ��	��	���$�'�0�0�:�0�$������������������������ûлܻ����ܻӻлû��Y�H�D�E�M�Y�f������ּ���༽�����r�Y�`�=�0�1�G�f�y�����н�������Ľ����`ED�D�D�D�D�D�EEEE*E7ECEEEAE7E/E*EE¿�r�h�i�o¦¿��������������¿�r�h�f�a�f�m�������������������������r�ּ̼ռּ������������������ìæàÞßàìù����������ùììììììD|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DyDtD|��	���(�5�7�<�:�9�5�4�(����������������������ùܹ�� ����ֹܹй¹������'�� �'�3�@�A�@�=�3�'�'�'�'�'�'�'�'�'�'�^�W�S�U�aÇÓàìðù������ùìÓ�z�n�^���
��������
��#�0�0�<�?�@�<�0�#���ݿҿϿѿҿݿ������� ��������� 5 l m � B B P = - , 6 4 l 8 I P T m  ` c 1 d J ) ' F   0 2 P  2 + A R A b ' = @ l a > r [ T * 9 P B g R 1 P Y , 5 M \ , x = | 8 E V 7    U  �  �  T  x  g  T  w  J  �  0  U  �  �  4  5    �  �  V  �  �  Z  
  �  �  Z  /  �  �  �  W  �  d  >  �  ;  K  �  �  �  ^  �  �  �  �  \  6  �  �  �  �    �  �  �    �  }  a  \  �  A  g  �  "  �    �  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D      *  3  ;  @  D  F  F  E  A  <  4  *      �  �  �  o      �  �  �  �  �  Q    �  �  {  C    �  h  D  3  �  �  �  �  �  �  {  n  d  Z  P  E  =  5  .  &              I  F  D  A  ?  <  :  7  5  3  -  %          �  �  �  �  �  �  �  �  �  r  X  7    �  �  �  f  9    �  �  �  �  h  �  �  �  �  �  }  y  s  m  g  `  Z  T  L  B  8  .  $        ?  >  9  0  &  '  )    
  �  �  �  �  p  T  C  '  �  T  &    �  �  �  �  �  �  v  _  F  +    �  �  �  _  -   �   �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  s  `  M  ;  �    C  S  _  �  �  �  �  �  �  �  e    �  q  �  @  2  �  �  )  a  n  m  h  a  U  H  8  $    �  �  �  |  [  :  %    �  �  �  �  �    c  ?    �  �  �  u  F    �  �     �  
  �  �  �  �  w  h  d  f  i  ^  Q  C  .      �  �  �  �          "             �  �  �  �  �  �  �  �  �    p  f  e  e  e  c  \  U  O  H  A  :  2  *  !        �  �  �                          �  �  �  �  �  �  �  �  �  *  <  ?  @  W  �  �  �  v  L    �  J  �  !  �  �  Y  �  �  �  �  �  �  w  i  Y  I  9  (      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  2  �  �  v  0  �  �  *  �  0  �  x  0  �  �  �    _  b  J  '    �  �  �  ?  �  K  �   �  �  �  �    ~  z  w  t  m  b  W  L  +  �  �  �  q  H    �  �  �  �  �  �  �  �  �  q  V  8    �  �  �  y  ;  �  �   �  �  �  �  �  �  �  �  �  �  �  �  t  e  R  8       �   �   �  t  ]  F  /       �  �  �  �  z  _  E  *    �  �  �  �  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  [  $  �  �    �  �  �  �  �  �  �  z  f  Q  :  !  	  �  �  �  �  �  �  8   �  �      �  �  �  �  �  �  `  0  �  �  �  U    �  &    �  �  m  Z  I  8  #    �  �  �  �  �  ~  d  E     �  �  �  ^    �  	t  
  
�  4  �  �  �  �  �  �    
�  	�  	9  -  �  &  �  7  C  K  L  K  C  5  #  
  �  �  �  �  m  \  I  ;  1  /  .    �  �  �  �  m  >  �  '    �  �  �  �  �  Z  �  \  �    a  \  V  P  J  E  @  ;  6  2  3  :  B  I  Q  \  i  v  �  �  A  1  !    �  �  �  �  �  p  P  .  
  �  �  �  k  @    �  3  (    
  �  �  �  �  �  �  �  �  �  �  �  |  f  M  3    ~  �  }  s  s  H    �  �  �  �  �  �  G  �  �  N  �  �  !    =  W  e  k  h  _  S  @  #  �  �  �  c    �  /  �  �   �  9  3  ,  )  #      �  �  �  �  �  �    F    �  j  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  8    �  �  O  ~  o  Z  >  !    �  �  �    S  $  �  �  u  Z  @    �  [  �  �        	  �  �  �  �  �  �  �  K  �  �  r  4  �  �  �  �  �  �  �  �  �  s  O  "  �  �  `    �  a    �  >  �  p  a  L  1    �  �  �  �  �  j  J  ,    >  S  5    �  �  H  A  ;  4  +  "      �  �  �  �  �  |  �  �    0  9  B  �  �  �  �  �  �  �  �  �  �  �  �  t  e  U  >  '    �  �  �  �  �  n  Y  C  -    �  �  �  �  �  ~  b  A    �  �  e  �  �  	N  	&  �  �  g  2  �  �  �  J    �  s    J  �  �    B  ?  :  !    �  �  �  {  N    �  �  o  *  �  �  .  �  �  �  �  �  �  i  O  9  +      �  �  �  �  �  �  ~  j  z  �  p  o  k  c  Z  Q  F  <  2  %      �  �  �  �  �    v  m  �  �  �  p  [  E  /      �  �  �  �  �  p  D    �  �  �      
    �  �  �  �  �  �  �  �  �  �  p  N  	  �  ^   �  �  �  �  �  �  �  �  �  r  T  7    �  �  {  7  �  �  l  V  �  �  �  �  �  �  e  G  *    �  �  �  �  �  �  �  �  �  }    �  �  �  �  z  r  v  �  �  �  t  a  I  +    �  �  �  �  %    
  �  �  �  �  �  �  �  �  �  �  �  q  [  D  /    
  �  �  �  �  �  c    �  �  w  :  
�  
I  	�  �  B  i  ^  �  �  1    �  �  V    �  �  s  Q  $  �  �  �  �  u    �  '  4  z  r  Z  9    �  �  V  	  �  b    �  P  �  �  |  g  K  &  �  	  	�  	�  	�  	�  	�  	�  	W  	  �  O  �  w    �  d  �  �  �  &  i  k  Z  D  -    �  �  �  �  b  3    �  �  R    �  =  .      �  �  �  �  k  F    �  �  �  _  %  �    �  �  }  �  �  �  �  �  �  �  �  q  V  8    �  �  �  |  S  )  �  �  	$  	  �  	  �  �  ]  #  @    �  �  =  �  �    �  �  L  �  �  �  �  �  ]  5    �  �  k  0  �  �  e  '  �  �  Z  A  L  �  �  �  �  j  ;  	  �  �  f  /  �  �  u  C      �  Y   �  �  �  �  �  �  �  �  �  w  g  \  U  M  F  >  6  .  '      �  �  a  8    �  �  R    �  �  N  '    �  %  �  �  �  �  �  �  �  �  �  �  t  l  e  T  A  -    �  �  �  �  d  4    �  �  �  t  X  F  .    �  �  �  @  �  �  /  �  8  �  �  