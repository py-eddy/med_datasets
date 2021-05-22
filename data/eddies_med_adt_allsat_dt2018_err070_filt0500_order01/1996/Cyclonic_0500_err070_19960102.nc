CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��j~��#       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N   max       Pՙc       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =Y�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @F!G�z�     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @v�\(�     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @O�           �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�`�           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       =<j       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�=   max       B.s�       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�n'   max       B.��       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�J�   max       C���       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C��       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Q       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N   max       P��A       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��O�M   max       ?�*�0��       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =Y�       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @F!G�z�     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�\(�     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @M�           �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�!-w1��   max       ?�*�0��     �  Zh                        &   7                     H      Q         %   (      
   G   !      .   (               
      	      %         &      	      
      
               
      <                     
   	      
         N�yXNT�SO)�=O;�N��~O�5�N)\�P�O���O(�NO�P-�AO��sO1_PՙcN2�ZP���N^.�O0rhPc��O��?NO�9N���P��@O�S0O3Z�O�T�Pp�Nc�8OPweN_S�O=+N�9O���O��O,��O� INa�O�E�OO�}O�S�N�
�O�oGNыxN+cN��N���O]�N`:XNs�aO��Nf�]O��O�N��NO�QN���N�/�NEcAN�ĨO�NN���NZ)rNM1�OZ7l=Y�=��;D����o��o�o���
�o�t��t��t��D���D���e`B��o��C���C���C���C���t���1��j�ě��ě����ͼ��������������������o�o�t��t��t��t���P�����#�
�#�
�#�
�''<j�<j�D���D���D���]/�aG��e`B�q���q���u�y�#��7L������㽟�w���w���
���T���^5��������
����������:<HRU`XUOH</::::::::���������������������� �������46<BOO[_fb[OIB864444v�����������������wv	 
	D[gt�����������pPECD#+/<HU^eifaUH</%#NOS[_htwy}��xth[ZPON}���������}}}}}}}}}}��������������������6IO[_wyw{thOB3#(���'1551)�������������������������z����
������zoz�����������������6t������hQJ6)497)).6ABHFB6)())))))))9;<HTamz|znmgaTQLD;9����#AObn{��zb<��CJUaz������|na[UTNDC��������������������V[dhkty~{tmhf[PSRVV,Hanz��������znU<#,_mz�������������m_Z_������ �����������)DKO[bhxqph[O6����6AIJD6��������#/9</#����������������������������������������
 #006<><0#
	�"������25N[_TPPRYTB51/1/,02tx������������vtqqrt��������������������/<HU\ZUKHD6#
� ,/<HJIHB</.-,,,,,,,,����������������������������������������18BN[ht��������tZN71���������������������������������))69BCGEB60)"��������������������!#07;:0/#!!!!!!!!����������������������������������������NNR[ddb][NNNNNNNNNNN������������������������������������������������������������BN[gqsl[B=5)ln{}�������{njfggkml�������������������>BN[gjg][NB@>>>>>>>>GHQUWabilleaVUNHHFGG:<@HIKPRQOKI><;204::��������������������rt{���������vtprrrr�����������������������������������������������������������������������������������������������".+&�����y¦«¦¦ŭŬũŭŰŹ����������Źŭŭŭŭŭŭŭŭìçà×ÔÓÐÈÌÓàìöù������úùì�n�m�g�f�e�n�{ŇŔŖŠŦťŠśŔŇ�{�n�n�a�]�U�P�T�U�_�a�n�zÆÂ�z�v�n�f�a�a�a�a�U�O�Q�8�/�)�U�a�n�z�{�ÈÆ�z�n�e�h�a�U�	��������	���"�%�"��	�	�	�	�	�	�	�	���������y�����Ŀѿݿ��
�����Ŀ������M�A�4�1�-�/�6�A�M�f�s�����w�q�f�c�Z�M�r�e�Y�N�L�F�B�L�Y�e�r�t�~���������~�v�r�3�-�,�1�3�@�E�@�@�<�3�3�3�3�3�3�3�3�3�3�m�f�`�Z�T�M�I�T�`�m�y������������y�m�m�m�C�;�4�G�T�y�����Ŀѿۿ��ѿĿ����y�m�Z�N�I�C�N�O�Z�g�s���������������{�s�g�Z�������#�,�/�:�<�H�L�H�A�=�:�/�#��'�����������6āĚĦľ��ķĘčČā�h�'F=F<F6F=FDFJFOFVFbFVFQFJF=F=F=F=F=F=F=F=�s�o�f�i�s��������"�O�a�m�v�m�d�H�"�	�s������������������������������������������	��������������(�5�>�:�5�(����������������g�N�9�3�6�A�Z�g�������������g�Z�P�L�Y�g�s�����������������������s�g�s�n�g�_�g�s���������s�s�s�s�s�s�s�s�s�s�����������������ĽнڽݽݽݽٽнĽ��������:����T�h�y�z��������������������������������������#�0�?�F�0�-�#�������ؿ.�%�$�$�*�.�;�G�T�X�`�k�m�o�m�`�T�G�;�.�عҹι������������ùܹ�� ��� ����ؼּͼܼ����!�.�:�B�J�H�@�:��������6�5�+�0�6�B�L�K�F�B�6�6�6�6�6�6�6�6�6�6�M�B�8�;�=�@�M�O�Y�f�h�r�q�t�u�t�r�f�Y�M�*�(�������*�3�6�7�6�+�*�*�*�*�*�*�����������������������üƼƼƼ����������׾ϾʾǾʾӾ׾ھ������	��	�����׾��������������������������������������������������������Ⱦʾ׾پ��׾ʾ������������������ʾѾ׾����ؾ׾ʾ������������������������Ŀſѿݿ���ݿѿĿ�������#�$�/�8�4�/�#���������`�S�G�T�`�y�������Ŀɿ̿ǿͿοĿ����y�`��(�5�N�P�W�S�N�J�A�:�5�(�����
���ʾ��������ʾվܾ�����	�����	���׾ʾ����޾׾ӾҾ׾������� ���������ܻû��������������������ûܻ������ܻx�l�l�_�_�\�_�k�l�x�����������������x�x����� ������������������������ûлۻܻ�ܻлû��������������������������������������������������������	���������������	��"�/�4�;�:�0�"��������������������������������������T�T�T�a�i�m�z���������z�m�a�T�T�T�T�T�T�#��������#�+�0�9�<�@�>�<�8�0�#�#�\�U�R�\�h�uƁƁƇƁ�u�h�\�\�\�\�\�\�\�\�U�J�@�9�,�#�%�/�<�H�[ÇÓßÞ×Æ�n�a�U���������'�4�@�M�P�W�X�M�@�4�'�����ɺǺ��ºɺֺܺ����ֺɺɺɺɺɺɺɺɽĽ��������ĽŽнѽԽҽнĽĽĽĽĽĽĽ�F1F,F$F#F$F0F1F=FJFVFcFeFcFcFVFLFJF=F1F1�������������(�4�9�A�B�A�4�(���ìåàÓÒÓàìù������ÿùìììììì����ĿĻĺĿ���������������������������̽�����������(�.�3�(����	�����ĦěĚčĉčĚĦĩĳĸĳĦĦĦĦĦĦĦĦ���������������
���$�(�&�$�$�������G�E�:�7�:�G�S�`�l�r�o�l�`�S�G�G�G�G�G�G�H�G�<�1�/�%�/�<�G�H�Q�J�H�H�H�H�H�H�H�H�ּԼѼʼǼ¼żʼ������������ڼ� 1 f N   K j i F + ( j 0 < H j - 7 h ! > a @ V  V V " S W K Q ^ 7 a ~ T  Z = 1 + v T \ $ ` ' ? ! m ] = d \ p ; ` D [ ^ < A l ] m f 5    �  v  �  N  �  �  _  �    C  M  7      �  �  K    a  �  �    R    a    |  �  �  �  �  �  W  �  �  �  j  �  k  .  �  X  �  �  �  F  z  �  �  t  �  6  �  ~  �  �  �  �  �  �  �  +  Q    �  �  �=<j=+��t��ě���C����#�
�8Q콃o��w�e`B�ě��'�`B�o��9X��/�Ƨ��
�ě��ixս}��h�C��\�q���]/��hs��+��w�aG��\)�L�ͽ8Q�]/�8Q�m�h���P�#�
��O߽��-��%�D����hs�L�ͽL�ͽaG��T������T���ixս�o�q����F��\)��7L��o���
��Q콺^5��-��-�� Ž�^5��-�������B��B:�B�yBx�B�/B!kA�=B	�fB[B�wB<�B�B��B�B��B%�B3�B�B��A�MB&=B�B)�B�B~�B 1 Bw�BkKB.s�B��B!��B��B%eB�.B��BݧB!��B�B��B+��BN�B	��B�B��B�B;fB%��B�BB�%B;qB-�B��B��B(�PB#2 Be�B��B&�sBF�B
g�B��BgmB�,B28B�B#sB��B�B�bB@BF�B ��A�n'B	�BǆB�XB�RB�RBԊB?�B��B��B?�B��B��A��GB&K�B��B@B�4B;5A���BFWB`B.��B�|B!B�BB$�5BT�B�|B�wB!�qB��Ba�B+?�B�KB	�ZB�,B�fB�B|�B%D�B ӣB>�B�WBO[BS B�BD'B(��B#?�BL�B�FB&K�B>mB
@
B�<B@wB�wB �B?mB?�A��NA��A�j�A�5�A��)A�A��Ax��A=�Y?�D�?���Aj��AqgA�?A_A�c$C���A�(�A�֣A��A���A�84A���A&#�A���A��Ae �>�J�A	}�A��@�lA�/@� AU�qA�MVAM��AN�Ax��A�U�Aq)ZA��=AU�EAU֥@�.�@�.�A��@��"A�8�A���A��fA��3A�lBg�A�=m@�K@<h,A'JC���A50�A̐A�;A2c�A��B�AhAÝ~AhpA���A��_A̅A�A��A�X�A�y>Av�+A=�?���?��Aj�fAp{�A�`A�y`A�1�C��A���A��A�F�A��~A�y�A�y�A&�ZA���A惨Ae�>��A�A׃�@�z�A��@�-cAT��A��IAKp�AN�aAy�A���Am�A�rAY�AU�@��@�:�A�~�@���A��A�z�A�G�A��A���B�A�~�@��@;��A'C���A7,4A�{YA�[�A2�cA��B�$A��A�|�A�v                        &   8                     H      Q         %   (         G   "      .   )   	            
      
      &          &      	      
      
               
      =                     
   
            	                     #      +               /         A      G         9   %         M   %      %   )                                 %      '      #                              #                                                               #               %         ;      !         9            #            )                                 %      %      #                                                                     N�yXNT�SN��gO;�N_c4O&�RN)\�O�5�O(ԷN��NN���O�@N���O1_P��AN2�ZO�(�N^.�O0rhPc��O��0NO�9N���O��OFFO GGOg�Pp�Nc�8OA��N_S�N���N�9Oe[�O��N�}>OGX�Na�O�E�OO�}O��N�
�O�oGNыxN+cN��N���N|��N`:XNs�aO��Nf�]OլN��EN��NO�QN���N�gNEcAN�ĨO�NNi�7NZ)rNM1�O?̌    C  �  w  �  M  h  j  	�  =  �  V  �  �  �  8  O  �  �  �  
  �  G  �  m  f  �  �  �  �  P  �  �  �  t  �  �  j    �  7  s  �  �  �  �    �  �  �  |    q  	n  �     �  �  $  |  ~  =    �  q    s=Y�=��o��o���
�t����
�T���ě��T���t��T����t���1��o���ͼ�C��T����C���t���1��`B�ě��ě��e`B��P��`B�,1�������o�o�\)�t���P�t��,1�0 Ž����#�
�,1�#�
�''<j�<j�D���e`B�D���]/�aG��e`B�� Žy�#�u�y�#��7L���P���㽟�w���w���
��{���^5�������
����������:<HRU`XUOH</::::::::���������������������� �������8BOY[b[SONB=88888888��������������������	 
	HN[g�����������|tSIH*/7<HIUX^^^UHE</.%%*V[fhktt{�uth[SPQVVVV}���������}}}}}}}}}}��������������������$*6BOhqtrth[OB0)-.-$))-,)�����������������������������zz�������������������06BO[hnrplid[OB:5//0)).6ABHFB6)())))))))9;<HTamz|znmgaTQLD;9����#AObn{��zb<��HNUanz�����znaZXSQIH��������������������V[dhkty~{tmhf[PSRVVKUant}������znaUHCCKhmz����������zzmgach������������������)6AO[hnnjjhe[O6)����6AIJD6��������#/9</#����������������������������������������
#+047:0#
�"������25BNSPPPRWNB5202/,12tx������������vtqqrt��������������������
#/<HFC=</#
,/<HJIHB</.-,,,,,,,,����������������������������������������5BN[gr��������t\SN:5���������������������������������))69BCGEB60)"��������������������!#07;:0/#!!!!!!!!����������������������������������������NNR[ddb][NNNNNNNNNNN������������������������������������������������������������)5<@>:51)nnv{}�������{nlhijnn�������������������>BN[gjg][NB@>>>>>>>>GHQUWabilleaVUNHHFGG:<AIIIOQQOKI=<<315::��������������������rt{���������vtprrrr���������������������������������������������������������������������������������������������&,*%������y¦«¦¦ŭŬũŭŰŹ����������ŹŭŭŭŭŭŭŭŭàÛØØÝàìðùÿ��þùìàààààà�n�m�g�f�e�n�{ŇŔŖŠŦťŠśŔŇ�{�n�n�a�U�X�a�e�n�y�z�{�z�r�n�a�a�a�a�a�a�a�a�U�P�H�D�<�7�<�E�H�U�a�n�s�x�{�q�n�g�a�U�	��������	���"�%�"��	�	�	�	�	�	�	�	�����������������Ŀѿݿ���������Ŀ����A�9�4�2�4�4�=�A�M�Z�f�n�u�s�n�h�f�Z�M�A�Y�R�L�K�L�Y�Y�e�r�~������~�r�e�Y�Y�Y�Y�3�-�,�1�3�@�E�@�@�<�3�3�3�3�3�3�3�3�3�3�m�i�`�[�T�O�P�T�`�m�y����������}�y�m�m�m�`�N�J�Q�h�y�����Ŀӿ׿׿ҿĿ��������m�g�\�Z�X�Z�g�g�s����������s�g�g�g�g�g�g�������#�,�/�:�<�H�L�H�A�=�:�/�#��6����������6�[āčġĵĨč�}�h�[�O�6F=F<F6F=FDFJFOFVFbFVFQFJF=F=F=F=F=F=F=F=�����}�}�������������������������������������������������������������������������	��������������(�5�>�:�5�(����������������g�N�9�3�6�A�Z�g�������������g�Z�T�P�T�\�g�s���������������������s�g�s�n�g�_�g�s���������s�s�s�s�s�s�s�s�s�s�����������������ĽнڽݽݽݽٽнĽ������T�>�4�4�;�H�a�m�z�������������������z�T�������������������
����
�
��������ؿ.�,�&�'�,�.�;�G�T�W�`�j�n�m�`�^�T�G�;�.��ܹ׹ѹù������ùϹܹ���������������ּͼܼ����!�.�:�B�J�H�@�:��������6�5�+�0�6�B�L�K�F�B�6�6�6�6�6�6�6�6�6�6�M�C�@�9�<�=�@�M�Q�Y�f�q�o�s�t�t�r�f�Y�M�*�(�������*�3�6�7�6�+�*�*�*�*�*�*�������������������������üļü����������׾ϾʾǾʾӾ׾ھ������	��	�����׾��������������������������������������������������������Ⱦʾ׾پ��׾ʾ������������������������žʾ׾۾ܾ׾ʾþ����������������������Ŀѿҿݿ��߿ݿѿĿ�������#�$�/�8�4�/�#���������`�S�G�T�`�y�������Ŀɿ̿ǿͿοĿ����y�`��(�5�N�P�W�S�N�J�A�:�5�(�����
���ʾ������;׾߾�����	�����	����׾ʾ����޾׾ӾҾ׾������� ���������ܻû��������������������ûܻ������ܻx�l�l�_�_�\�_�k�l�x�����������������x�x����� ������������������������ûлۻܻ�ܻлû������������������������������������������������������	�� �	�	���"�/�0�/�"���	�	�	�	�	�	�������������������������������������T�T�T�a�i�m�z���������z�m�a�T�T�T�T�T�T�#��������#�+�0�9�<�@�>�<�8�0�#�#�\�U�R�\�h�uƁƁƇƁ�u�h�\�\�\�\�\�\�\�\�a�]�U�N�N�R�U�a�n�zÃÇËÇÇ�z�u�n�a�a��������'�4�@�M�M�S�R�M�@�4�'���ɺǺ��ºɺֺܺ����ֺɺɺɺɺɺɺɺɽĽ��������ĽŽнѽԽҽнĽĽĽĽĽĽĽ�F1F,F$F#F$F0F1F=FJFVFcFeFcFcFVFLFJF=F1F1��	�����������(�4�8�A�A�A�4�(���ìåàÓÒÓàìù������ÿùìììììì����ĿĻĺĿ���������������������������̽�����������(�.�3�(����	�����ĦěĚčĉčĚĦĩĳĸĳĦĦĦĦĦĦĦĦ����������
���� ����������G�E�:�7�:�G�S�`�l�r�o�l�`�S�G�G�G�G�G�G�H�G�<�1�/�%�/�<�G�H�Q�J�H�H�H�H�H�H�H�H�ּԼʼɼżɼּ�����
���������� 1 f >   R F i A  $ j , < 8 j ! 7 C ! > a : V  3 Q ! U W K N ^ - a w T   ; = 1 + w T \ $ ` ' ? = m ] = d  E ; ` D Z ^ < A l e m f %    �  v  �  N  �    _  ?  a  �  M    -  �  �  �  K  �  a  �  �  �  R    �  �  X    �  �  �  �    �  L  �  �  �  k  .  �  �  �  �  �  F  z  �  �  t  �  6  �  5  �  �  �  �  �  �  �  +  Q  �  �  �  �  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�    y  t  l  a  U  I  >  3  +  #      �  �  �  �  �  �  �  C  <  5  /  '          �  �  �  �  �  �  �  �  �  �  �  K  ]  g  ~  �  �  ~  o  a  Q  ;    �  �  Y    �  |  /  �  w  f  U  Q  R  ?    �  �  �  R    �  r  '  �  o  �  ?  �  �    P  y  �  �  �  �  �  �    m  V  ;       �  �  �  �  �  �  �    1  I  8      �  �    	  �  �  �  `  ?      h  ^  T  J  A  A  A  A  <  0  %      �  �  �  �  �  {  c  (  R  f  g  X  A  '  	  �  �  �  �  ]  .  �  �  )  �    �  �  	g  	�  	�  	�  	�  	�  	�  	�  	\  	  �  f  �  ]  �  �  �  �  �  �    7  :  *      �  �  �  �  b  2  �  �  A  �  _  �  J  �  �  �  �  �  �  �  �  �  �  |  n  ^  L  :  (      �  �  U  V  T  N  C  4  "    �  �  �  �  k  N  :    �  �  �  G    R    �  �  �  �  w  p  k  U  :    �  �  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  `    �  f    �  q  P  -    �  �  w  :  �  �  z  <  (  �  �  6  �  g  �    +  7  3    �  �  �  �  �  v  R  +  �  �  r  
  �     �  O  R  V  S  N  B  1    �  �  �  y  K    �  �  �  T     �  �    !    '  _  �  �  �  �  �  �  `     �  �  {  �    �  �  �  �  �  |  v  p  g  ]  R  H  >  3  (         �   �   �  �  |  p  e  U  D  3      �  �  �  �  �  j  P  5     �   �  
  �  �  g  \  B  4  )       �  �  �  �  �  l  4  �  z   �  �  �  �  �  �  �  �  x  f  K  )  �  �  �  I  �  �    <  W  G  E  B  @  >  :  7  3  )      �  �  �  �  �  g  '   �   �  �  �  |  t  m  f  _  Y  P  C  3       	    �  �  �  �  �    g  �  �  �  �  /  \  k  h  R  2    �  �    �  �  �  �  �  =  R  U  N  @  N  W  <      %    �  �  }    �  �  �  �  �  �  �  l  M  *  �  �  �  r  3  �  �  [  "  �  �  �  �  |  �    >  _  u  �  �  w  W  )  �  w  �  ,  �  \  �  [  �  �  �  �  �  �  ~  X  (  �  �  �  v  J    �  �  C  �  '  B  �  �  �  y  l  ^  Q  D  5  %      �  �  �  �  t  =  �  �  J  P  L  B  5  %    �  �  �  �  �  �  f  3  �  �  �  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    y  r  �  �  �  �  �  �  �  �  m  V  9    �  �  S    �  �  >  "  �  �  �  �  �  �  �  �    y  t  p  k  g  d  `  \  V  K  @  I  r  a  L  8     
  �  �  �  n  ,  �  �  H      �    �  �  �  �  �  �  �  �  �  �  p  L  &  �  �  �  �  f  8     �  S  f  v  �  �  �  �  �  z  f  G    �  �  u  +  �  �  9  �  f  B  �  i  g  _  O  <     �  �  �  K    �  P  �  �  �          �  �  �  �  �  �  �  �  �  }  \  ;     �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  Z  ,    �  �  �  G  �  v  7  �  �  �  �  �  �  �  �  X  "  �  �  N  �  �  W  �  }    m  r  r  o  h  [  K  4    �  4    �  �  �  @  �  �  +  �  �  �  �  �  �  s  Y  @     �  �  �  _  J  6  	  �  �  g  /  �  �  �  �  �  m  q  a  _  e  \  ?    �  �  �  �  �  �  �  �  �  �  �  �  �  q  V  8      �  �  �  �  �  �  |  M    �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  q  i  `  W  O    	  �  �  �  �  �  �  �  �  �  �  �    x  p  e  [  O  B  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  8  �  �     �  �  �  �  n  W  @  '    �  �  �  �  �  u  \  D  ,     �   �  |  t  l  d  \  T  L  H  F  D  B  @  >  >  A  D  F  I  L  O          �  �  �  �  �  �  �  �  x  c  J  2    �  �  �  q  h  ^  U  K  B  8  0  )  !          �  �  �  �  �  �  3  S  ~  �  	  	b  	b  	Y  	b  	l  	O  	  �  f  �  �    �    �  �  �  �  �  �  �  �  �  �  �  n  R  5    �  �  �  d  4       �  �  �  �  �  �  �  �  �  p  [  M  ?  +    �  �  �  `  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  d  T  D  5  %  �  �  �  s  ^  E  )  	  �  �  �  m  9  �  �  i    �  n      "    �  �  �  �  i  I  *  
  �  �  �  �  [  1  �  �    |  n  c  V  F  6     �  �  �  d  0  �  �  y  9  �  �  v  4  ~  z  w  r  m  e  ]  R  F  4    �  �  �  �  V    �  �  ]  =  4  +  !        �  �  �  �  �  �  �  �  �  �  �  �  �      
  �  �  �  �  �  �  �  x  E    �  �  �  �  j  I  (  �  �  �  �  v  �  �  �  �  �  d  �  �  �  �  x  z  �  �  �  q  d  X  K  <  !    �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  }  a  E  '  	  �  �  �  �  f  >    �  �  �  p  l  p  `  J  5      �  �  �  e  (  �  �  s  -  �  �  |