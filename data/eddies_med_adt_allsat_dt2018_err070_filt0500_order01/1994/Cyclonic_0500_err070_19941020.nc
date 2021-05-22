CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�M����       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Y   max       P��[       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       � Ĝ   max       <�`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?}p��
>   max       @F�33333     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v��Q�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @Q            �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�%�           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �(��   max       ;��
       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�W�   max       B58P       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z~   max       B5:Q       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?$1�   max       C��T       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?/�   max       C��;       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Y   max       P���       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��+j��g   max       ?ڗ�O�;e       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �!��   max       <��
       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�        max       @F�33333     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�z�G�     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @R�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�:`           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z�G�{   max       ?ڐ��$tT     �  ]            	         -               �   	       8         S            }      0         u            "      "               y   	      
      
                  
      	   .         	   
            "            %                  Ou�8N���N��N���Op��N�|�O�,N�HiNQ��N��P|P���O�O��+P:&�O?`6M�YO�C�N��UOeFN�X�P�� N&�=P�YN<
3N�T�P��[N�S�O+�N�NO�?�N��oO��Nڸ�P%�#O7;NvPl)N|YNON�`�N�2O+��N O�yM�(O_��N[#�Nr{�OAg�N��P+
O<�O$�_N�Ne�WO��O�ځN���O�SO��mO{��N��O�<=N�C�OG�*Ol2�N�N?	-N���<�`B<t�<o;ě���o��o�D����o��`B��`B�t��t��#�
�49X�D���e`B�e`B�e`B��C���t���t���t����
���
���
��9X�ě����ͼ�������������/�+�+�C��C��t���P��P��P���#�
�#�
�',1�,1�<j�<j�<j�<j�@��@��H�9�P�`�T���Y��]/�e`B�m�h�u��+��O߽�hs��hs���T����-� Ĝ� Ĝ��������������������HNS[gght{|�tg[NMHHHH��


������������dht������thbddddddddanz���������znhaZY[annz����������~znnnnnz������������ztqoprz�� 	
 #&%##
�������������������������')6BGOORONB6)!�������������/>Knz����ynaH<#	�����������������~��������
���������������������������Y[g����������tgfe_]Y#(/40/#"o{|������������xtroo����������������������������������������������|���������������yrs|�%��������������������������������������������������//8<HU\^[WUH<;5/////��nU/��<H�������]agmpstqpnmgaa`ZYZ[]vz}���������������zv�()))"����y|��������������|yxyhd\TODC:=CKOZ\dhihhh #/?HUVOHC9/#"#+//01/,##y�������
�����qpy��������������������()66=96)$&((((((((((��+/2;:?LG</#
�����	
!
							#,06950#��������������������RTajlmnnmma_[XTTRRRRT_`afms{~}zpmaTSNMOTMO[`hiha[[ONMMMMMMMM����)������Z[gty}tgf[ZZZZZZZZZZz�����������������zz��������������������(),+)!)6BO[htvwqjh[SIB6)!��������������������#$7HUz�������znUH<+#����������������������#).)�������������������������z{������zvwzzzzzzzzz{�����������}uqpquv{� 
#<IUbhfbUJ<#
�������������������������������������� #"�����"#'-<INQPMI<0}��������������~}}}}����������������������������������������")5CNX[t}���tgNB5'"��������������������bgkstvwttjgabbbbbbbb")35754))(>BCNW[[\\[XRNEB=>>>>����������������#�5�<�H�P�H�<�/�#��
�������������������������������ѿͿ˿ѿݿ����ݿѿѿѿѿѿѿѿѿѿѼM�I�A�A�M�P�Y�`�c�c�]�Y�M�M�M�M�M�M�M�M�A�>�@�A�C�N�Z�d�g�s�����������x�g�Z�N�A����������ÿ�����������������������������z�v�u�yÇÓàìü����������ùìàÓÇ�z�;�3�.�"�!��"�&�.�;�G�H�R�T�T�X�T�G�;�;���z�t�����������������������������������_�[�S�Q�K�P�S�_�_�l�w�x�{�v�s�x�{�x�l�_�����	�� �5�A�b�s�w�y�z�u�g�Z�A�(�E�E�E�E�E�E�E�E�E�E�E�FVFqFzFtFgF@FE�Eٿ;�7�.�+�$�,�.�.�;�G�T�U�[�`�d�`�T�K�G�;ÝÕÓÇ�n�a�a�nÀÇÓàäìó����ùìÝ�(�	����A���������ȾǾɾǾ������f�M�(�����������������������������������������%�)�6�6�7�6�)�����������û��������������ûܼ��%�,�(�����л����������
���$�&�0�2�0�/�$�����������������B�R�[�g�q�g�[�S�N�B�5�)����z�¦«­¦�3�"��'�8�Y�~���ֺ��!�O�-� ���⺤�~�Y�3�y�o�m�i�l�m�y�����}�y�y�y�y�y�y�y�y�y�y���ۿڿ����(�N�g���������g�N�(����������������������������������������������A�A�4�1�,�-�4�A�M�Z�a�e�[�Z�M�A�A�A�A�A�������������������m�M�J�B�C�T�z�������������������(�5�6�A�N�O�N�A�5�(�����ݿѿĿ����������ĿĿѿݿ�������꿸�����������������ĿпѿۿѿοĿ��������Ŀ������y�u�x�����Ŀݿ���(�0�(����ѿĿy�y�y�m�i�`�^�T�G�E�G�I�T�V�`�j�m�y�y�y�H�#��
���
��4�<�U�_�n�zÉÐÇ�n�a�HàÛÔàììù������������������ùìàà����
�������"�/�C�V�]�Y�U�V�H�"�	�����������������Ǿʾ׾������׾ʾ���ùøùù������������ùùùùùùùùùùD�D�D�D�D�D�D�EEE7ECESE[E[EUECE*EED��@�7�4�(�4�@�M�Y�Y�f�g�f�Y�M�@�@�@�@�@�@����������������������b�a�Z�a�b�n�s�{łł�~�{�p�n�b�b�b�b�b�b�h�]�]�h�u�|ƁƎƚƞƥƚƎƁ�x�u�h�h�h�h����Ƴƭ����������������	� ���������ɹ�߹߹�������������������������������������������� ����������������� ��������������������������ļĹĳĬĳ���������������������0�$�$���$�'�0�=�>�I�L�I�=�0�0�0�0�0�0�A�5�9�A�C�N�Z�e�c�Z�W�N�A�A�A�A�A�A�A�A�s�h�r�~�������������������������������s�)�'�#�)�5�=�B�E�N�[�_�e�[�N�B�5�)�)�)�)���������������������ʾ׾�������׾���	����������$�0�=�@�=�0�$����������������������������ʾ׾׾ʾ����������3�2�1�3�@�K�L�Q�V�L�@�8�3�3�3�3�3�3�3�3�����������ĿɿοƿĿ����������������������������лܼ�'�@�[�`�X�M�@�4�'���ܻû��:�!��
�	��!�)�-�:�F�K�N�S�V�_�]�S�F�:�����������������ɺֺٺֺܺɺɺ��������������������ͼܼ�����%�(�"�����ּ����Ľ����������������нݽ��������߽нĽܽн̽ݽ������(�4�A�M�F�(�������ŭŢŠŕŔŇ�ŇŔŠŭŹŻ��żŹŭŭŭŭ�<�0�#��
��������
�#�;�K�Z�n�v�x�n�U�<�������������!�'�(�'��������t�j�p�t�{āăĈčĔĚģĦİĮĠěĒā�tčĂ�t�q�j�j�h�m�tāčĚĥģğĞĤĠĚč���������������������
������������������ÓÎÇÂÇÑÓ×àìíìàÜÓÓÓÓÓÓ�z�q�n�b�a�a�a�n�zÂÇÓÖÓËÇ�z�z�z�z e B / O . \ $ = M R / 4 6 ^ T 0 B X ) \ V g x d f ) O ~ . 3 \ v V Y n D 9 , i C N } M F = n z m D � S D � 8 c L v V @ e G \ Q Y 0 � P � : J  _    *  �  �  �  +    |  9  W  3  5  }  �  �    \  �  "  �  �  �  �  f       �  v  �  �  �  �    E  f  9  �  ~  )  �  �  �  =  m  L  �  �  �  	  �  �  �  }  M  �  �  �  �  ~  w  ;  �  x  �  L  �  L  Q  �;o;D��;��
�o��9X�t��D����o�T����9X�ě���㼛��,1��7L���ͼ���ě���1��w���n����ͽ�7L��9X�,1�\)���C��+�q����󶽉7L�P�`�ixսD���@�� Ĝ�<j�#�
�D���<j�L�ͽ@��ixս49X�m�h�]/�e`B��o�aG���j�y�#��+�y�#�}󶽩�罛�㽛�㽾vɽ�{������-��/�ȴ9���ͽ�;d��vɾ%�T�(��B��B	�B�BjDBUiB��B ��B�BF+B�QBU�Bv�B�HB �B �&B
�B�,BR�B�B2B��B�!B��B:B�BAB>�A��	B �FBb�B*R�B1hB'�B�_B��B58PB{�B�B$�B%�~BxA�W�A���B7vB&�B	Z�B�B2�B��B4^B|�B|�B��B��B�?B��B)ރB%�B §B-�]B�B%��B
�B�B�WB�XB�|B	�,B��BM�B��B	�B�tB��BE�BƑB ��B$OB?�B�B42B�dB�B��B >B
7�B�B?�B�mB�B��B�XB�}B@�B�BB{B�A�z~B ��B^jB)�B1<rB?�B��B,B5:QBBcB��B$�B%��BGA��tA��BBmBՉB	�yB��B
�^B��BA�B��BI�B��BH�B��BêB)�4B%}bB ��B-̵B@B&C�B%�B�pB��BN�BSRB	@7B��B?�A�n?A���A|U^@�A�A��A�x�A��Ac?A�@��aA�W�C��TAd�A��BAB#1A���A�i�@�8�B	6�A��A��?�Z�Am:'A�	�A�FfA;`�A��A�њA{|�Aw��Az��AiZA�
NA��pA�=�AQ��A��^C�w�@Ղ@A���B2%B\�?$1�A�1�AWj�A���B
i1A�OA�q&A��VAO��B��AK??��(Awk�@���@yʴ@.:A��A'�&A3d�A��~A욐@���A�3�A�G�A�WlA��YAȯ�A�}uA�ehA|�I@�{�A���Aπ�A�dAb��A�Z@�A�pZC��;Ad�A�y�ABA���A�|)@�6�B	j�A�/A�y�?��wAm�A��A��A; A�
fA���AyiOAw)NA{'Ah�A��Ä́cA��AP��A�w�C��@�X@�A�yTB��B��?/�A���AXt�A�z�B
��A��A�r�A�i�AO�B��AJҺ?�#�Aw35@�V�@s�@,|AvA(�A6�A�l.A�v@��A��A�~dA���A�r2A�            	      	   .               �   
       8         S            ~      0         u      	      "      #               y   
                           	         	   .         
   
            "            &                  	                                 #   A      #   /         %            C      -         E            +      #      /         '                                          +               +         !      !      %                                                   #   9         -                     !      +         9            )      #      '                                                                  '                     %                  O�5N���N��N���OO�CN�|�O�,N�fNQ��N��P|P���N��N˾�P$��N�M�YOzRN��UOWձN�X�O���N&�=P�zN<
3N��dP�$�N�S�O+�N�NO��N��oO�N�Nڸ�P	�O7;NvO�\9N|YNON�`�N�2O�{N O�yM�(O_��N[#�N@@�N�k%N��OS�N�X�O؎N�Ne�WO� O�ځN��Oe\pO��mOm!�N��O�<=N�C�OG�*Ol2�N�N?	-Nd܉    �  �  �      	}  �  c  �  �      
  �  �  �  	�  �    �  
p  �  a  �    	(  4  �  �  �  �  s  �  5  s  x  ;  �  >  �  3       �  �  �    G  c  �  �  �      �  �  ?  �  S  �  j  �  �  c  �  
  <  u  �<��
<t�<o;ě��D����o�D�����
��`B��`B�t�����49X���ͼ�o��C��e`B�C���C����㼓t����P���
��1���
�����P�`���ͼ���������`B��/�C��+���C��t����
��P��P���#�
�'',1�,1�<j�<j�@��L�ͽ@�����T���Y��T���Y��e`B�e`B�q����%��+��\)��hs��hs���T����-� Ĝ�!����������������������HNS[gght{|�tg[NMHHHH��


������������dht������thbdddddddd]ahnz��������znba][]nnz����������~znnnnnz������������ztqoprz��
#$$#"
�������������������������')6BGOORONB6)!�������������/=Ranz����naH<#������������������������������������������������������������egt��������tphgeeeee#(/40/#"~��������������{utu~�������������������������������������������������������������������%��������������������������������������������������4<<HUY\YUH<<44444444
/U�������nZUH/#	
]agmpstqpnmgaa`ZYZ[]vz}���������������zv�()))"����z~�������������~yyyzhd\TODC:=CKOZ\dhihhh  #/>HMUUNHB8/##+//01/,##xz�������������}x��������������������()66=96)$&((((((((((��
"',-+)#
������	
!
							#,06950#��������������������RTajlmnnmma_[XTTRRRRTTakmpxz|{zmaTTONPTTMO[`hiha[[ONMMMMMMMM����)������Z[gty}tgf[ZZZZZZZZZZz�����������������zz��������������������'),*)";BO[hstutqmh[WOLEB;;��������������������./2<HU^ailmiaUHD<7/.����������������������� &#�������������������������z{������zvwzzzzzzzzz{����������~vrqrsww{� 
#<IUbhfbUJ<#
�������������������������������������� #"�����#&+0<IMPPMI<0 }��������������~}}}}����������������������������������������")5CNX[t}���tgNB5'"��������������������bgkstvwttjgabbbbbbbb")35754))(@BEN[[[[VQNHCB@@@@@@�
�������������
��#�(�/�9�<�=�<�/�#��
�����������������������������ѿͿ˿ѿݿ����ݿѿѿѿѿѿѿѿѿѿѼM�I�A�A�M�P�Y�`�c�c�]�Y�M�M�M�M�M�M�M�M�N�E�A�@�A�C�H�N�Z�g�s�~�������s�r�g�Z�N����������ÿ�����������������������������z�v�u�yÇÓàìü����������ùìàÓÇ�z�;�6�.�"�"��"�+�.�;�A�G�Q�S�T�W�T�G�;�;���z�t�����������������������������������_�[�S�Q�K�P�S�_�_�l�w�x�{�v�s�x�{�x�l�_�����	�� �5�A�b�s�w�y�z�u�g�Z�A�(�E�E�E�E�E�E�E�E�E�E�F=FkFvFqFdF=FE�E�EͿ;�/�.�'�.�.�2�;�@�G�T�T�Y�^�T�G�;�;�;�;ìêàÜÚ×ÓÑÓÖàãìíùþûùîì�4���	��A�M�����������þľ������f�M�4��������������������� ����������������������%�)�6�6�7�6�)�����������û����������ûлܻ�����!����ܻл����������
���$�&�0�2�0�/�$�����������������5�B�N�[�g�g�[�R�N�B�5�)����z�¦«­¦�L�@�4�7�@�E�L�Y�r���������������~�e�Y�L�y�o�m�i�l�m�y�����}�y�y�y�y�y�y�y�y�y�y���ܿڿ޿����(�N�g�������g�N�(����������������������������������������������A�5�4�.�0�4�A�M�Y�Y�N�M�A�A�A�A�A�A�A�A���z�f�S�L�P�a�z�������������������������������������(�5�6�A�N�O�N�A�5�(�����ݿѿĿ����������ĿĿѿݿ�������꿸�����������������ĿпѿۿѿοĿ��������Ŀ����������x�}�����Ŀݿ���&�����ѿĿy�y�y�m�i�`�^�T�G�E�G�I�T�V�`�j�m�y�y�y�H�#��
���
��#�7�?�U�]�n�zÇÆ�n�a�HàÛÔàììù������������������ùìàà�"���������"�/�<�P�X�V�P�R�H�/�"���������������Ǿʾ׾������׾ʾ���ùøùù������������ùùùùùùùùùùD�D�D�D�D�EEE*E7ECEMEPEJECE<E*EEED��@�7�4�(�4�@�M�Y�Y�f�g�f�Y�M�@�@�@�@�@�@����������������������b�a�Z�a�b�n�s�{łł�~�{�p�n�b�b�b�b�b�b�h�]�]�h�u�|ƁƎƚƞƥƚƎƁ�x�u�h�h�h�h����������������������
��������������̹�߹߹�������������������������������������������� ����������������� ��������������������������ļĹĳĬĳ���������������������0�$�$���$�'�0�=�>�I�L�I�=�0�0�0�0�0�0�A�8�:�A�F�N�Z�`�Z�Y�U�N�A�A�A�A�A�A�A�A�����������������������������������������)�'�#�)�5�=�B�E�N�[�_�e�[�N�B�5�)�)�)�)���������������������ʾ׾�����ܾ׾ʾ��������������$�0�'�$�������������������������������������ƾ������������3�2�1�3�@�K�L�Q�V�L�@�8�3�3�3�3�3�3�3�3�����������ĿɿοƿĿ��������������������������ܻ��'�@�Y�]�Z�V�M�@�4�'����ܻл��:�!��
�	��!�)�-�:�F�K�N�S�V�_�]�S�F�:�������������ɺֺ׺ںֺɺǺ��������������ʼ��ļʼѼּ߼�����"�&� ������ּʽĽ����������������нݽ��������߽нĽ�߽ӽݽ�������(�4�@�L�E�(������ŭŢŠŕŔŇ�ŇŔŠŭŹŻ��żŹŭŭŭŭ�<�0�#��
��������
�#�;�K�Z�n�v�x�n�U�<�������������!�'�(�'��������t�j�p�t�{āăĈčĔĚģĦİĮĠěĒā�tčĂ�t�q�j�j�h�m�tāčĚĥģğĞĤĠĚč���������������������
������������������ÓÎÇÂÇÑÓ×àìíìàÜÓÓÓÓÓÓ�z�u�n�d�h�n�z�~ÇÓÕÓÇÃ�z�z�z�z�z�z R B / O - \ $ > M R / 1 6 j U # B J ) Z V 1 x ^ f 0 Y ~ . 3 T v Q Y n D 9 3 i C N } = F = n z m C K S  � $ c L u V ? U G X Q Y 0 � P � : L  _    *  �  �  �  +  �  |  9  W  �  �  <  5        �  �  �  �  �  R  f  �  �  �  v  �  =  �  �    �  f  9    ~  )  �  �  D  =  m  L  �  �  c  #  �  �  �    M  �  }  �  �  �  w    �  x  �  L  �  L  Q  n  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  �  �  �  �      
  �  �  �  �  �  [    g  �  �    F  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \  5    �  �  �  �  �  �  �  �  �  �  �  p  _  O  B  5  '       �  �  �  u  e  T  C  1      �  �  �  �  �  �  �  �  j  K  ,        �  �  �  �  �  �  �  �  �  d  =    �  �  I     �    
  �  �  �  �  �  �  �  �  �  �  o  X  B    �  e     �  	}  	D  	  �  �  Q    �  y  (  �  ^  �  �    �  �  $  L  c  ~  �  �  �    z  q  d  U  F  6  %       �  �  �  ~    �  c  T  E  6  %      �  �  �  }  b  F  +     �   �   �   �   x  �  �  w  g  U  A  .      �  �  �  �  �  m  ,  �  l     �  �  �  �  z  Z  7       �  �  �  �  �  �  �  �  �  �  9   �  J  
    �  �  `  �  n  K  �  W  �    
�  	�  �  �  g  C  �          	    �  �  �  �  �  �  �  �  }  _  <     �   �  �  �  �  �  �  �  �  �      �  �  t  ?    �  �  �    &  �  �  �    s  l  c  L  /    �  �  �  �  �  9  �  O  �    �  �  �  �  �  �  �  �  �  �  �  �  �  n  Y  C  )  
      �  �  k  R  9      �  �  �  �  {  `  D  (     �   �   �   �  �  	  	i  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	u  	  �  �  �  _  4  �  �  �  �  �  �  �  �  �  �  �  �  q  a  Q  H  B  <  6  0        �  �  �  �  O    �  �  l  /  �  �  �  �  ^    �  �  �  �  �    `  <    �  �  �  {  g  B  �  �  A  �  �    �  	�  	�  	�  	�  	�  	�  
  
i  
i  
K  
  	�  	R  �    1      u  �  �  �  �  �  �  �  �  �  �  u  d  P  <  (        �  �  K  _  U  <    �  �  �  �  j  6  �  �  i  �  d  �  #  �  p  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  �    
            �  �  �  �  �  ~  F    �  �  �  g  6  �  �  	  	!  	'  	#  	  �  �  �  u  F    �  5  �  �  �    j  4  &      �  �  �  �  �  �  �  n  U  <  "    �  �  �  �  �  �  �  �  �  v  d  S  <  $  	  �  �  �  �  R      �   �   �  �  �  �  �  �  �    t  i  [  M  >  (    �  �  �  �  �  e  �  �  �  �  �  �  s  Z  <    �  �  �  m  3  �  �     �  #  �  ~  q  c  U  H  :  -        �  �  �  �  �  �  �  �  �  g  k  N  ;  !  �  �  �  �  |  �  �  �  �  �  �  r  ?  �  �  �  �  �  v  \  @  &        �  �  �  �  ]  &  �  �  u  5  !  %  .  4  4  )    �  �  �  z  ?     �  �  �  T  5    �  s  g  [  M  >  -      �  �  �  �  e  P  8    �  �  b   �  x  u  r  o  i  b  W  L  B  7  )  -  <  (    �  �  �  �  �  �  !  �  ,  �  �  '  7  !  �  �  V  �  �  �  }  	�  �  �  m  �  �  �  �  o  ]  K  9  '      �  �  �  �  �  8  �  ~  D  >  >  >  >  ?  ?  ?  <  7  1  ,  &  !      �  �  �  �  �  �  �  �  �  �  �  t  c  O  7      �  �  �  I  0     A  h  3  )        �  �  �  �  �  f  >    �  �  �  j  /   �   �  �        �  �  �  �  �  �  z  `  C  $  �  �  �  H   �   �     �  �  �  �  �  �  �  �  �  �    l  Y  D  0      �  �  �  �  �  u  g  Y  F  0      �  �  �  v  D    �  y  <  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  x  o  f  \  S  I  @  �  �  }  {  t  l  b  H  %  �  �  �  �  �  m  [  Y  �  �  �    (  9  1      �  �  �  �  �  �  �  \  5    �  �  �  d  (  8  F  A  9  )      �  �  �  �  �  �  �  y  ^  A  �  �  O  @  1  8  Z  U  C  4  #    �  �  �  �  Y    �  �  I  �  �  �  �  �  w  g  S  ?  )    �  �  �  �  �  c  /  �  �  ^  �  �  �  �     x  �  �  �  �  {  L    �  p    �  �  �  �  �  �  3  �  �  �  R  �  �  �  �    �  �    &  C  `  |  �  �             �  �  �  �  �  d  +  �  �  n  +  �  �  E        �  �  �  �  o  F    �  �  x  =    �  �  E     �  �  �  �  �  �  j  S  <  %    �  �  �  �  �  �  �  _  <    �  �  �  �  r  N  )  &  '  -    �  �  �  ]    �  y  �  o  ?  %    �  �  �  �  �  �  �  y  `  G  -    �  �  k  #  �  �  �  �  �  �  �  �  }  h  T  <    �  �  x  3  �  �  P  �    J  M  9  "      �  �  �  l  8  �  �  O  �  t  �  K    �  �  �  �  �  �    b  E  '    �  �  �  {  I    �  '   d  d  g  [  N  >    �  �  �  �  [  &  �  �  b    �  g  �  �  �  �  �  n  Z  F  1      �  �  �  �  k  G  $     �  �  �  �  y  \  A    �  �  b  ,  �  �  L  �  t    �  ?  �  =  `  c  [  R  G  8  %    �  �  �  �  a  =    �  �  �  �  V  B  �  �  |  O    �  �  L  �  �  9    �  �    �  k  &    �  
  �  �  �  �  _  @    �  �  {  h  V  8  �  �     �  M    <  "    �  �  �  �  �  q  b  N  6      �  �  �  �  �  p  u  f  V  G  8  (      �  �  �  �  �  �  z  ]  C  )    �  �  �  �  �  �  �  a  B    �  �  �  j  R  :  #    �  �  �