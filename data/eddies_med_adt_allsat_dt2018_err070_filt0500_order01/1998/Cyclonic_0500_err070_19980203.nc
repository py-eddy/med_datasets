CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�������       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nm   max       P��Z       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��-   max       =0 �       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @Fk��Q�     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
<    max       @vq\(�     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @4         max       @R            �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��            5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       =o       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0u       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B/�       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >P   max       C�t�       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@J2   max       C�k�       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          e       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Nm   max       P�@O       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e��ڹ�   max       ?�s����       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��-   max       =0 �       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��z�H   max       @F.z�G�     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vqp��
>     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P            �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @�.�           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?~�Q��   max       ?�p:�~�      �  Y|            P      d                   	                  	                           &      '   0            +                  @   5      
            (            	            G                  	               
Nh>oN��{O��gP��O�^;P��ZP �N���O3�Ol��Ozc�N�ҮOM��O��dPޡO+�N�F�N�q/O>H�N�B�N��N��HOE�)N��_NZW�N���O�ɫN�V�O���O�Z	O��zN(�$O�rLPICOf�^O� zN?�MOG�sOi�O��P7�NO}eN+ŲOS{|O�/�Nn�.OO%]NY1�N���N8o�O8�Oa��O%�OgsP{�NC=<N��Ox�N,DQNmN��1O��Ol�jN�@"O���O�=0 �<���;D��%   �o�t��#�
�49X�u�u��t���t����㼛�㼴9X�ě�����������`B��`B��h��h��h��h���+�+�C��C��\)�\)�\)�t��t���P��w��w�#�
�#�
�0 Ž0 ŽD���D���H�9�H�9�P�`�P�`�T���Y��]/�]/�aG��m�h�q���q���q���u��������O߽�����㽝�-���
���-{��������{{{{{{{{{{{)67=>:6)'{����������������{x{��#0�������{I<#���)5;BFJLIB5)�����������������|�)EN[glkltkgYIA%|�����������~�||||||�������������������@HMQUWYbhhkhaUH<656@MO[hkt��������thXOFM��������	

�����������������������������
#*35(
�����6BO[ht����{t[YB6+-06[_hmt���������th[UU[��������������������%))+.354,)'#/<AHNOPPNH</#��������������������qzz��������zspqqqqqq�����������������������������������

����������������������������GHPUaica`VUHC=GGGGGGnz����������zouvwrmn;<EHUZansnmbaWUHB<:;)8BShpnpoh[O5)hbL8	*6CQ\���vjh�����������������������������jm�������������zhbbj�����������������������������������������������������#*0<DH><0/#INZgt���������tg][GI��������������������OUnz����������n[QNO��������������������RTahmrwxvqma_VTSPPRRY[ehtw���th[YYYYYYYY{�������������zuoqz{��������������������imuz����|zqmmmffiiii���������������������������������������//<FHMIH@<3/$)+-////=BNO[aa][OFB========DITUbnqqpngdbYUIIBADdt}��������������thd��������������������{����������������{z{�
&.+/6/#�����#$+)%#������	


����������������������������������������������+5BNONGB50++++++++++>BJNO[agigf`[WNJBA>>�����������������������������������������������5:NU[gp�ztg]VNB5,,/5W[]^aggt������thg[WW��ټ޼�������������������ֺͺϺպֺ�����������ֺֺֺֺֺ��U�O�N�=�2�0�<�H�U�a�n�{Ã�n�i�j�q�n�a�U�������l�X�M�(�&�N�s��������������������������x�e�W�Z�b������������������������ù×ÄÈáù����B�hčĩĦĚĆ�r�)������ݿĿ��������ѿ����"�(�5�A����������������������	������������������������������������!� �����������������������������������������������¿³³³¿������������ �������ؼf�d�Y�M�@�?�@�F�M�M�Y�f�h�r�u�u�}�r�f�f���������$�0�6�=�H�K�H�?�=�0�$���ýùìåææäçìù�����������������žھؾ޾������"�;�G�Y�c�`�Z�O�.�"��ں�������'�3�@�L�L�B�>�;�9�;�3�'��H�F�B�?�<�A�H�R�T�`�X�U�T�J�H�H�H�H�H�H�s�r�f�_�f�s�����������������������s�s�Z�W�M�I�G�K�M�U�Z�f�s�����������s�f�Z�f�a�^�Z�W�T�M�M�M�Z�f�o�s�}�|�{�x�s�f�f���������������$�)�,�$��������׾˾ʾ������ľʾ׾۾��������׾׾׾׺r�h�e�^�]�d�e�r���������������������~�r�ɺǺ��������������ɺκֺݺ���ֺѺɺɼY�S�T�Y�Y�f�r�|�x�r�f�b�Y�Y�Y�Y�Y�Y�Y�Y�a�^�T�Q�M�T�a�j�m�z�{�z�z�m�a�a�a�a�a�a���������������	�"�;�@�D�F�;�/�"�	�������	�������������	���"�$�$�"���	���������x�r���������ûлۻ߻׻û��������z�`�G�;�2�)�����������;�G�Q�`�m�zƧƚ�u�T�\�]�d�uƁƎƚƳ��������������Ƨ�g�a�Z�N�P�Z�g�g�o�p�g�g�g�g�g�g�g�g�g�g����ĿļĿ�����������
�"���
����������I�0�#� �(�:�b�nŇŔŠŲŽžũŔ�{�n�U�I�}�}�����������ÿĿѿܿܿؿҿſ��������}�׽˽ǽŽĽݽ������(�+��������׾����
���(�)�(�(����������	���������������	���"�,�,�'�"� ��	������׾־Ͼ׾������	�
����	����ù������������ùܹ�'�1�'������ܹϹú��Y�B�L�O�Y�e�l�r�����������Ӻ����������������������������������������������ù������������ùŹȹ̹͹ùùùùùùùÿ��������ѿ޿�����������ݿ׿ѿĿ������N�C�E�K�N�[�j�s�t�v�g�[�N�5�/�(�$�(�)�5�6�A�E�N�Z�[�Z�N�A�5�5�5�5����������������)�6�B�H�O�I�B�6�)��л˻ɻ̻лڻܻ���߻ܻллллллл�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������������������������������������������������������ûȻܻ����ܻܻлû��������������������������"������������#�+�0�<�A�I�N�S�I�<�0�#��ŹŭŗŔŇ�{ŇŔŠŭŹ����������������ŹE*ED�D�D�D�EEEPEiE�E�E�E�E�E�EuE\ECE*E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������/�-�,�/�1�;�H�Q�T�V�]�a�a�a�a�V�T�H�;�/�Z�N�Z�a�g�s�|��s�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�Ľ����½ĽƽнҽӽнĽĽĽĽĽĽĽĽĽ������������������������
��
���������������������!�&�(�&�!���������������$�0�=�V�\�e�e�b�V�I�8�0�$�����S�M�G�B�A�G�S�`�l�p�y�|�y�s�l�`�S�S�S�S�h�a�B�<�3�,�/�<�H�K�[�nÇØáÝÓÇ�z�h��������ĿĳĲīīĳĿ������������������ . < c U I d > 9 ; G D D 6 + ( x � r . g @ Z F ? p ^ l R H r ] Q 4 c W 8 m ] f J ? K k f < m F j ^ K 1 u 3 B � S : G ` X B U Y . } M  {  �  �  �  �  P  n  �  �    �  �  �  ;  j  �       �  6  �  �  �  �  �  �  p  2  �  �    S  K  �     z  �  �  L  l  ]  M  �    Z  �  �  �  �  k  3  P  e  �  
  [  �  Y  J  d  �  w    �  �  @=o<#�
��`B������P��;d�,1������`B��w��h�����H�9�8Q�''o�\)�]/�o�49X�#�
�<j�t�����㽏\)�@��������T�����}󶽡���}�q���<j�q���]/����ě��y�#�ixս����O߽aG���Q�u����m�h��%�}󶽋C���1������}󶽲-��hs��t����9X�ě��Ƨ���ě�B�@B5hB �B&�rB�0B�&B*BLB�PB��B"'B#v�B��B�:B�B'�B'�BV�Bi�B!FLB �BB�B"r�B#��B��B�dB��B��B��B0uB~BB�B [sB�/B�B"!B%��B	��B�@B}zB��A��B�B{B�A�]^B�B^�B�bB�YB'sB
��BRlBzB��B�&B��B-Bd�B��Bo(B-�	B=�BSWB�B	�eB6_B>�B ��B&��B��B�B=�B[�B�fBI�B��B#�=B�GB;�B�kB �B��B@NB?�B!�CA��AB?�B"� B#RdB >VB�9B�B�B��B/�B?BL�B @aB@�B��B"%B&:�B	ŕB�>B�[B��A��BƼBʡB��A�d�B�9B?�B�B�6B'?�BABA�B?�B?MB��BTB��B5�B�SBI4B-~�B��B?�B��B	��A�f@D�A�\A��fA��]A��A�5�A�N#A��nA��kA��5@�?�B	��AΩ�A^D?���A�d�AG�AAn9A@�B	 AS]�@Za@3�@�jA��A��A[9�@�YkA\��Bb�A��$A�k�A�nAu�A-�A4�oA\�AW�m>��\@L�A��q>PA{)wA���A�s�Aե8@�6�C���A��@��8A�qOA��A�&8C���C�t�A�-.A�j�A��lA'A���APCB
D�AL�A�6A��A�@D nA�c�A��1A���AוPA�O�A�|oA��TA��vA�|�@�[7B
>Aψ�A^^?���A�q�AI !A@ƎA@�*B	@�AS�@3c@5+r@�0A�	RA��bA[�@�zfA[ B=�A��.A�3A�k�Au\A-_A4�rA[BAV�>��@A��>@J2AyhA�70A�~A��@��KC��.A�@1@��+A�oA�`IA�t/C��C�k�A���A�~�A��7A'�A�y�A�]B	E�A�A�~�A⣯             P      e   !               	                  
                     	      &      (   1            ,                  @   6      
            (   	         
            H                  
               
         !   E   !   G   )                        %                                    )      #   ,   !         +      !            '   /                                          /                              !               ?   !   C                           %                                                !         +                     -                                                                        !   Nh>oNUt�N�]�P��iO�^;P�@OO�|�NK�xO3�OvOzc�N�ѵO-`8O�^�O�\�N�*N�F�Ne,�O0FN�B�N}�IN��
OE�)NwK�NZW�N���OHZ N���O8�[OO��zN(�$O}��PICO%
N��9N?�MOG�sOi�O�F�P0�8N���N+ŲOS{|N�o�Nn�.OO%]NY1�N���N8o�N՚Oa��N��MOgsOG��N,I(N��O=�N,DQNmN��1O��Ol�jN��8O���O�  �  �  4  �    
+    v    �  k  �    g  �    (    �  �  �  �  F  �  a  �  V  4    �  �    �  j  �  �  �  �  �  �  *  �  �  �  �  �  	�  �    d  �  �    >  C  �  �  �  �    v  �  �  l  �  �=0 �<�9X�u�ě��o�T����C���o�u��9X��t����㼴9X��1���ͼ�/������/��h��`B�o����h�����+�<j�t��@��]/�\)�\)���t��0 ŽD����w�#�
�#�
��%�8Q�H�9�D���H�9�q���P�`�P�`�T���Y��]/�aG��aG��u�q����1�u�u��+�����O߽�����㽝�-���T���-{��������{{{{{{{{{{{)16986)����������������������#<n�������p<0���)5;BFJLIB5)�����������������~��$)0?KN[efhnng\NCA5-$���������������������������������������;<BHU\_abebaUH@<::;;MO[hkt��������thXOFM�������

 ��������������� ���������������
#)24.'
�����26BO[hw���t[OB>6.02[[ht�������th[YW[[[[��������������������())-133)#/<@HLNOOH</$#��������������������tz�������zurtttttttt����������������������������������� 

 ����������������������������GHPUaica`VUHC=GGGGGGwz������������zyxvtw;<=GHUajja_UUHF<;;;;5=BGO[dhhjh_[OBA3+,5*6CFMOQTOC6*�����������������������������lz������������ztjddl������������������������������������������������������������#*0<DH><0/#INZgt���������tg][GI��������������������]huz����������zni]W]��������������������TTaemrvwupma`WTSQQTTY[ehtw���th[YYYYYYYY{�������������zuoqz{��������������������imuz����|zqmmmffiiii���������������������������������������//<FHMIH@<3/$)+-////=BNO[aa][OFB========GIUbnonmiebb_ULIDDGGdt}��������������thd��������������������{����������������{z{� 
#%'$"!
���#+)$#������	


���������������� ������������������������������+5BNONGB50++++++++++>BJNO[agigf`[WNJBA>>�������������������������������������
����������5:NU[gp�ztg]VNB5,,/5W[]^aggt������thg[WW��ټ޼�������������������ֺӺҺֺܺ��������ֺֺֺֺֺֺֺ��H�H�A�@�H�U�`�a�e�c�a�U�H�H�H�H�H�H�H�H�������s�\�I�6�1�:�Z�s����������� ���������������x�e�W�Z�b������������������������ùàÇËäù����B�hāčĤęă�o�)������ݿѿĿ����Ŀѿݿ�����!�(�(������������������� ��������������������������������������!� ���������������������������
��������������������¿³³³¿������������ �������ؼr�j�f�Y�N�M�J�M�P�Y�f�f�r�t�t�x�r�r�r�r��	�	�����$�0�=�E�I�F�=�=�0�$�������ùìèçåèìù�������������������ž��޾�������"�.�;�K�W�Y�W�T�H�.�"��������'�.�3�=�9�5�4�6�3�'�����H�F�B�?�<�A�H�R�T�`�X�U�T�J�H�H�H�H�H�H������s�q�s����������������������������Z�X�M�I�H�K�M�W�Z�f�s���������~�s�f�Z�f�a�^�Z�W�T�M�M�M�Z�f�o�s�}�|�{�x�s�f�f��� �����$�&�*�$����������׾ξʾ������Ⱦʾ׾پ�������׾׾׾׺r�h�e�^�]�d�e�r���������������������~�r�ɺɺ������������ºɺֺ̺ٺ���ֺ˺ɺɼY�S�T�Y�Y�f�r�|�x�r�f�b�Y�Y�Y�Y�Y�Y�Y�Y�a�^�T�Q�M�T�a�j�m�z�{�z�z�m�a�a�a�a�a�a�������������������	�
���	���������������	����������	���"�"�#�"������������������������ûʻлѻϻĻû������������������	���"�+�-�-�*�%�"��	����Ƨƚ�u�T�\�]�d�uƁƎƚƳ��������������Ƨ�g�a�Z�N�P�Z�g�g�o�p�g�g�g�g�g�g�g�g�g�g����ĿĽĿ�����������	��
�	������������I�0�#� �(�:�b�nŇŔŠŲŽžũŔ�{�n�U�I�����������������ĿѿҿҿѿͿĿ������������ݽսнϽнսݽ�������� ����������
���(�)�(�(����������	���������������	���"�,�,�'�"� ��	������׾־Ͼ׾������	�
����	����ù������������ùܹ����������ܹϹú��r�Y�N�R�Y�e�o������ � ���Ѻ������������������������������������������������ù������������ùŹȹ̹͹ùùùùùùùÿ��������ѿ޿�����������ݿ׿ѿĿ������t�n�g�c�c�g�g�t�t�t�t�t�5�/�(�$�(�)�5�6�A�E�N�Z�[�Z�N�A�5�5�5�5����������������)�6�B�H�O�I�B�6�)��л˻ɻ̻лڻܻ���߻ܻллллллл�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��������������������������������������������������������ûлܻ����ܻٻлû����������������������������"������������#�&�0�<�>�I�K�J�I�<�0�#��ŹŭŗŔŇ�{ŇŔŠŭŹ����������������ŹE\EPECE*EEE*E:ECEPE\EiEuE�E�E�E�EuEiE\E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������/�.�-�/�2�;�H�T�T�T�\�a�a�a�`�V�T�H�;�/�Z�N�Z�a�g�s�|��s�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�Ľ����½ĽƽнҽӽнĽĽĽĽĽĽĽĽĽ������������������������
��
���������������������!�&�(�&�!���������������$�0�=�V�\�e�e�b�V�I�8�0�$�����S�N�G�C�B�G�S�`�j�l�x�q�l�`�S�S�S�S�S�S�h�a�B�<�3�,�/�<�H�K�[�nÇØáÝÓÇ�z�h��������ĿĳĲīīĳĿ������������������ . 0 G N I d + : ; . D M 1 ) & d � j ) g . T F D p ^ L C : < ] Q $ c L  m ] f ) 9 G k f < m F j ^ K + u 0 B c O : @ ` X B U Y 8 } M  {  l  �  `  �    o  d  �    �  �  s  $         �  u  6  �  �  �  �  �  �  �  �  �  m    S  �  �  ]  �  �  �  L    .  6  �    �  �  �  �  �  k  �  P     �  �  9  �  =  J  d  �  w    �  �  @  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  �  ~  s  g  \  P  B  2      �  �  �  e     �  Q  �  �  �  �  �  �  �  �  �  �  �  \    �  l    �  K  �  {        /  3  -        �  �    4  #  �  �  g    �  `  �  �  �  �  �  �  b    �  b  �  �    �  h    �  6  �  �   �    �  �  �  �  �  �  �  l  B    �  �  �  o  <  �  �  j  G  
  
*  
  	�  	�  	`  �  j  �  �  �  �  �  U  �  k      [  |    [  j  v    ~  w  j  X  E  3  #    �  �  �  >  �  k  �  �  �  �  '  g  p  t  t  j  T  A  -    �  �  �  �  y  <  �    	    �  �  �  �  �  �  �  �  p  \  E  (    �  �  P    �  %  p  �  �  �  �  �  �  �  h  E    �  �  w  =    �  H  k  b  X  I  8  #    �  �  �  �  �  �  �  �  z  \  :   �   �  �  �  �  �  �  �  �  �  �  �  v  g  X  G  5     	  �  �  �  �        �  �  �  �  z  V  /  �  �  i  	  �  &  �  ,  l  ]  e  e  `  U  A  #  �  �  �  6  �  �  L  *  	  �  �  �  �  �  �  �  �  �  �  �  �  ~  \  8    �  �  �  v  L     �   �  	          �  �  �    t  s  o  f  `  Y  I  *  �  �  �  (        �  �  �  �  �  �  �  �  �  �  �  |  q  e  Y  M  
        �  �  �  �  �  s  N  (    �  �  x  ;   �   �   s  �  �  �  �  �  v  x  l  P  0    �  �  q  +  �  z    �  �  �  �  �  �  �  �  �  �  �  �  |  m  _  P  A  +     �   �   �  �  �  �  �  �  �  �  �  l  O  0    �  �  �  ]  $  �  �  �  �  �  �  �  �  �  �  �  i  B    �  �  }  D    �  �  B   �  F  <  1  )  %      �  �  �  t  C    �  �  z  K    �  V  ~  �  �  �  �  z  q  g  ]  Q  D  7  &      �  �  &  o  �  a  W  M  A  4  &      �  �  ~  �  N  !  �  �  �  e  2     �  �  �  �  u  c  Q  @  ,      �  �  �  �  �  s  Z  B  *  �  �  �  �  �  B  R  T  F  0    �  �  @  �  ~    �  7  �        4  (      �  �  �  �  �  c  8    �  �  A  �  E  a  �  �  �  �        �  �  �  �  y  =    �  p    F    �  �  �  �    K  ~  �  �  �  ~  X  #  �  �  1  �  �  Q  �  �  �  �  |  c  I  -    �  �  �  �  �  �  n  @  �  6   �   K        �  �  �  �  �  �  �  �  �  �  �  �  �  }  r  f  [  S  �  �  �  �  h  >    �  �  T  �  �  ;  �  d  �  r  5  Y  j  c  R  8    �  �  n  .  �  �  ~  [  R    �  5  �  w  .  �  �  �  �  �  �  �  �  �  �  ~  O    �      �  Q  �  R    �  �  �  �  "  X  v  �  �  |  j  Q  -  �  �  �  ;  �  q  �  �  �  �  �  ~  p  d  X  I  ;  ,      �  �  �  �  �  �  �  �  �  �  �  �  p  J  %  �  �  �  �  o  N  2      �  �  �  �  n  Y  A  '    �    �  �  �  �  ]  "  �  �  c  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ;  �  �    (  �    *    �  �  P    �  _    �  C  �  p  �  q  �  g  �   �  �  �  �  �  �  �  �  p  X  ?  #    �  �  �  n  A    �  �  �  �  �  �  y  f  R  ?  .      �  �  �  �  o  N  ,    �  �  �  �  �  �  |  Y  2    �  �  u  G  ?    �  �  d  +  �  �  �  �  �  �  ~  |  �  �  �  �  �  �  �  e  4  �  �  9  �  �  �  �  �  �  �  ~  j  W  C  -    �  �  �  �  �  `  =    	�  	u  	@  	  �  �  R    �  �  O    �  �    �  �  Q  �  �  �  �  ~  f  M  6  "    �  �  �  �  �  �  �  �  s  l  q  v    �  �  �  �  �  �  �  }  `  B  #    �  �  *  �  Y  �  �  d  W  J  =  0  #      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  X  6     �   �   �  �  �  �  �  �  �  �  �  �  h  C    �  �  �  �  �  �  g  A  �  �  �       �  �  �  �  �  �  �  {  U  .    �  �  @   �  >  /     	  �  �  �  �  �  h  >    �  �  �  O    �  �  |  
  	�  	�  	�  
>  
=  
�  A    
�  
v  
  	�  	t  	@  �  �  |     �  j  �  �  r  ^  I  6  $    �  �  �  �  �  i  0  �  �  G  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  U  �  �    e  J  -    �  �  �  q  <    �  �  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
  *    
  �  �  �  �  �  �  �  �  �  m  Z  G  5  #        �   �  v  q  l  c  Y  L  ?  /      �  �  �  k  \  T  c  r  �  �  �  �  �  �  �  �  k  Q  7    �  �  �  �  �  Z  $  �  �  0  �  �  s  S  .    �  �  n  �  �  ~  O  	  �  D  �  �  "  �  ^  k  h  a  X  L  =  *    �  �  n     �    �  F  �  �  �  �  �  �  �  e  ?    �  �  �  `  ,    �  x  J    �  U  �  �  �  j  `  W  @  $    �  �  �  �  Y  -  �  �  �  �  �  }