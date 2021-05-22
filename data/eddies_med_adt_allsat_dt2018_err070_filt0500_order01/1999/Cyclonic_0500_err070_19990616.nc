CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���Q�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�   max       Pi�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�9X       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @E������       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @vx�\)       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q            �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @�R�           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �B�\   max       <���       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�Ht   max       B-<V       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�j#   max       B-�O       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =�>�   max       C���       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =s��   max       C���       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�   max       PJ@}       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��N;�5�   max       ?���IQ��       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�9X       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @E������       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vx�����       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q            �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @���           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���p:�   max       ?�~($x     0  ^      
                        -   @                  
                  	   	            $   Z         !   2   "   :                   
         7   4                                    7      �   	   .            #      '         	   N��;O?��N�*�M�NyDPP�YO4��NIIOZ��O��PeN~s NB��N�aND�	O�!No/O>�xNȷNM�~N�'{NM�O�N�/�N��O7�OY�ZPHPi�N9	�O$PJ@}P9��PR9{O�̲O�dO���N��NqȎOY@N��IO"c5OolrO��PRN��zN&AN���O��N�N�O�W9P�?N� �N�MTO3��PB�N;�qO�N�=�P?�O�J�N�2xO,X�O��O27�Oӂ�N
�}N%:�N�>[N5��<�9X<�1<D��<o;D���ě���`B�t��49X�D���T���T����C���C���C���t���t����㼛�㼣�
���
��1��1��1��9X�ě��ě����ͼ��ͼ��ͼ���������/��`B��`B�������o�+�C��C��\)�t��t���P��P��������w�#�
�#�
�'0 Ž0 Ž8Q�<j�L�ͽP�`�P�`�T���Y��]/�m�h�m�h�}󶽇+��\)������"/2204/"LNP[gq|�����tg[RMKLVamz}�}znmmaYUVVVVVV')68BHB?653)''''''''NO[ahqtwth[ROONNNNNN/<anz������}nRE<*)/#/<MQJHD></%#����������������������)*/)�������mqx���������������rm��!0Ui�����nUI0
��1<AINUbgbYUI<9111111X[\gilhg^[OOXXXXXXXXuz{���������������zu����������������������������������������������������������
��������� 

          FHIUZabaaUIHFFFFFFFF��$).)����������������������� )67BCO[^hkb[OB6.)# rty|�������������tr����������������������������������������4BFNO[`fjmnf[OB64214amywz����������zpj]a	)BSkqm[LH5#	#/7<=</%#����������������������#0{����mUF<0
�������������������������
"<YC0#
������#0<FLI<80#
����fgt������������}tmff����������������������������������������FHJUX]a^UHHBFFFFFFFF.6?LO[ho{}tqpi[OB6).������������������������������������������������������������	#/<ACGKPRH/#	Uanz�����������naYUU&)5BHBB55)'()669A6)%&((((((((((�������� �����������)5BN[u��}[B5)�����������������������	�������mqtx{���������{rnmlm������!�������������������������������		 ��������������������������������5>?5)�������+/;>>@@>;:212/++++++���/<>>=6-#
�����_amqzz|||zmmkca`____����
/:FH</���������������������������^git�������tig^^^^^^����������������������)5MRTNB5)���fht���������trjga_\f��������������������������������knz����znikkkkkkkkkk		�� }�������{}}}}}}}}}}���������
��#�,�%�#��
�����������������ѿȿĿ����Ŀѿڿݿ������������ݿ�à×ÔÚàìù��������ýùìàààààà�����������������������������������������e�b�a�e�k�r�}�~�������~�s�r�e�e�e�e�e�e�"�� ���"�/�H�T�a�z���������z�m�H�;�"�O�G�B�6�0�,�-�6�B�O�Q�[�h�s�r�h�a�f�[�O�<�5�/�+�/�<�H�U�a�b�a�U�P�H�<�<�<�<�<�<�����������������&���"���������U�I�<�0�*�-�6�<�U�b�nŇŔŠŬţŔŇ�b�U�ݽнƽ�������������*�?�C�;�(���ݼʼ����������żʼͼּؼ޼ڼּʼʼʼʼʼʾZ�X�Z�d�f�s������s�f�Z�Z�Z�Z�Z�Z�Z�Z�����������������������������������������(�)�6�6�6�)�������������������������������
������
��������¿²©²º¿�����������������������������������������������+�6�*����������������¼ʼ˼ʼ�����������������������������������������������������������	������� �	�	���"�-�"�"�&�(�"���*�(� �*�6�C�H�O�V�O�C�6�*�*�*�*�*�*�*�*������������������������������������������}�s�g�Z�Y�P�[�Z�Y�Z�_�g�s�x�����������U�a�g�m�n�n�n�k�a�U�H�D�<�8�3�<�H�T�U�UÓÏÉÈÌÑÓàìùú����������ùìàÓ��������}�s������������ɾѾվ;ʾ�������ƮƳ��������������$�0�5�6�/�#�������s�W�Q�P�Z�s������������	��#���������s������������
���
�� �����������������s�n�g�g�j�s���������������������������s�������������m�c�7�6�Z�s���������������𺗺r�Y�:�.�4�Y�������ֺ��"�:�7���ֺ������ɺƺ������ɺ���-�_�}�u�u�O�3�*���ɻܻû������������ûԻ����"�������z�w�y�x�{�����������������������������z�׾ʾ������������ƾ׾��������������z�n�n�d�n�zÇÇÈÇ�z�z�z�z�z�z�z�z�z�z�Z�O�N�F�N�Z�g�s�y�t�s�g�Z�Z�Z�Z�Z�Z�Z�Z�ܹ׹ù����������ùϹֹܹ����������ܺ���������������"�'�'�)�'�$���@�5�4�8�@�I�L�Y�c�e�r�s�~���~�r�e�Y�L�@������������������'�,�1�.�)�������E�E�E�E�E�E�E�E�E�E�E�E�FFF$F"FFE�E�{�w�|�������������ùϹ������ڹ����{ŭūŠŞŚşŠŭŭŹŹ��������Źŭŭŭŭ���������������������������������������ҿG�@�;�.�"�����"�.�;�<�G�J�K�G�G�G�G�߾ݾ�վѾоؾ������	������	��߻����� ����'�4�5�4�'��������������
�
�������"�*�6�:�7�6�*�"����'�����������'�@�Y�\�_�[�M�F�@�4�'�f�]�b�y���������%�%�����ּʼ�����f���{�y�p�m�l�m�y������������������������������������	���"�#�"��	���������������z�x�q�p�x�����������ûȻû�����������ÓÆÆÓèù���������� �'�"������ìÓ����������������������������������������ED�D�D�D�D�D�D�EEECEPEUETEREIE7E*EE�"� ��"�.�/�;�H�T�`�V�T�H�;�4�/�"�"�"�"�����������������Ŀѿڿ�� ��	����ѿ���ĿĻĿ������������2�9�C�E�<�#��������Ŀ�I�>�=�:�1�=�D�I�U�V�]�Z�V�S�I�I�I�I�I�I�������Ŀѿݿ�������������ݿڿѿĿ����{�h�W�G�?�C�Q�tāčĎċċēğħĩħč�{ččďēĚĦĳļĿĿ��������ĿĽĳĦĚč�Y�@�4�'�������'�@�f�r���������r�YàßÓÏÓàìðñìàààààààààà�����������¿Ŀȿ˿Ŀ��������������������!������!�.�/�:�@�G�M�G�:�.�!�!�!�!�����#�/�2�2�/�#���������� R %  z L F 1 l Y D 2 f 8 a d @ G T - C Z [ X Q 3 * 4 e U e , o � n B g 4 R G C ) 0 % 6 ` X I a n � ( > r Q @ 2 A � * n I H I D C 0 Z e h _ 6    �  �  �  E  �  �  u  �  �  �  �  �  h  g  �  4  i  �  8  R  �  r  b  	    �  �  �  /  �  \    /  �  ^  �  o  N  o  �    b  �  i  �  �  K  �  1  �  �      �  �  �  �  �  !  �  �  F  �  z  �  �  !  %  ^  �  G<���<49X;o;��
�#�
�C����ͼT����h�ixս��㼋C���9X�+��1����/�o��j���ͼě��ě������0 ŽH�9�49X�u��xռ�h��w�m�h���-�}󶽮{�0 Žu�\)��P����49X�H�9�P�`��^5��-��w�0 Ž0 Že`B�,1�8Q콁%��7L�<j�D����o���ͽP�`�B�\�q���ě����T�m�h�����j��t������hs��������A���B	(�A�I3B��BU�B#�BM�BG�B!�B ��B&��B&�?B�B�&B9�B�BC(B��B$Z�B�<B	�BB\�B9'B!b�B��B��BA3B�B�}B�=B&�GBZB$1�B$�TB
YHB�pB��B��B6�B!�B!7B{�B4BB�IB�VBN�B�	B rWB�zB)S(B-<VB*éA�HtB �B]`A��B�A�k�BI�B�B	ܷBnB<�B
�B#yB8/BC�B2\B
��A���B	3.A��B��B@�B�@BG}B7�B�;B?�B&�B&�wB�vB��B��BñB=�B2�B$DNB�8B��B=�B��BC�B!?�B��B��B<B�\B�(BB&��B��B#>�B%?iB
?hB>�BB� B?�B! -B!B�B��B��B�xB��B�Bo�B@B ��B?HB)>�B-�OB*��A�j#B 0�B?�A���B�mA��zB#�B?B
�BzB��B	ѺB>�BB�B٣BA�B
A�A��A}~7A�w�A�5]?��yA��fA�ZA�`WA�s�A��A/�b@�̒AB�A�A�L�A�\A�ߣA��a@�AA���A�y�B qA�nA��A�iMA��-AK�PB�XA�łA��EA�z|A��N@$TG@a�@��A��ASYA�:iA��>�|�?f��?ف(A���C���=�>�A��`A��Aa�AX�?@ÇxA���@˯J@�=AoCzA���@�0@AϢxA�JjC�s[A��uAzW
A��B?�A{�0A���A��T@��)A˫�Aw��A�iA�QA�E�A~A̒dA��?���A���AؒAÜdA��LA�(A0�g@��\AA��A�6�A�~�A�Z�A���A�o8@�AA���A���B  �A�s�A���A�o�A�mMAK�YB��A��A��A�t.A���@�x@U�@�-A�� AT�A�2�A�J�>�A?o��?ЉhA��PC���=s��A�|A�p�Aa��AW �@�1A�^@�ܺAҴAn�A���@�`�A�y�A�Z�C�o�A�.�A{ɏA�B4uA|Aۂ4A���@��OA���Ax��A�A�~�                              .   @                  
                  
   
            $   Z         !   3   #   :               !            8   4                                    8      �   	   .            #      (         
                     +            #   9                                                   -   7         9   7   ;   '      !                     !   +            %            7            )      #      '   %         '      '                              +                                                                              9   3   9                                             %            7            '            #            %      %            N��;O)��N�*�M�NXQ�O�Y�O4��NIIO2t�O��{O��qN~s NB��Nt|7ND�	N�
�No/O'��NȷNM�~NG�hNM�O�N�/�N��O1�OZaO�?"O�1|N9	�O$PJ@}P$��PIo�O��N��ROy��N��NqȎNVMPN��IO"c5OH"	O��O �KN��zN&AN���O��N�N�Ou� P�?N� �N�MTO*g�PW?N;�qO��N�=�O�n�Oql�N�2xO,X�O��O27�OȄN
�}N%:�N�>[N5��  �  O  �  0  �  U  r    �  �  �  �  G  �  �  /  �  �  f    \  L  �  �  ]      �  �    
  O  �  �  K  �  �  �  �  �    �  �  �  �  }  �  �  �  C       %  	  �  �  *  �  %  ,  �  �  �  �  .  �  �  �  �  �  g<�9X<��
<D��<o;o�t���`B�t��T������8Q�T����C���j��C����㼓t����
���㼣�
��1��1��1��1��9X��/��`B�o������ͼ�����������h�8Q�+�\)���o�L�ͽC��C���P��㽃o��P��P��������w�'#�
�'0 Ž49X�H�9�<j��1�P�`�e`B�y�#�Y��]/�y�#�m�h��%��+��\)������"/2204/"LNR[gotz|~tmg[SNNLLVamz}�}znmmaYUVVVVVV')68BHB?653)''''''''NO[_hosh[SOONNNNNNNN,4<Unz�����zvvnV@2-,#/<MQJHD></%#�����������������������	(�������y���������������|vsy"$08?IUbhkjh[UI<0)#"1<AINUbgbYUI<9111111X[\gilhg^[OOXXXXXXXX����������������������������������������������������������������������������

��������� 

          FHIUZabaaUIHFFFFFFFF�!����������������������������� )67BCO[^hkb[OB6.)# rty|�������������tr����������������������������������������47@BJO[aehha[POB<654x��������������zupnx)5HNOPKB5)#/7<=</%#����������������������#0{����mUF<0
��������������������������#XUKB0
������
#2:==;40)#
���lt����������tqllllll����������������������������������������FHJUX]a^UHHBFFFFFFFFLO[hlkh[[[OFLLLLLLLL������������������������������������������������������������	#/<@EFJOOH/#	`anwz������ztnhba`_`&)5BHBB55)'()669A6)%&((((((((((�������� �����������)5BN[u��}[B5)�����������������������	�������nrv{����������{snmmn������!�������������������������������		 ����������������������������������1<=3)������+/;>>@@>;:212/++++++���
#&.12/,'#
����_amqzz|||zmmkca`____���
0:<5+#���������������������������^git�������tig^^^^^^������������������������)5BKON5)��fht���������trjga_\f�������
�������������������������knz����znikkkkkkkkkk		�� }�������{}}}}}}}}}}���������
��#�,�%�#��
�����������������ѿ˿Ŀ����Ŀʿѿݿ��� ��� �����ݿ�à×ÔÚàìù��������ýùìàààààà�����������������������������������������e�d�b�e�l�r�~�����~�r�m�e�e�e�e�e�e�e�e�;�"��"�#��/�H�T�a�z�����������m�T�H�;�O�G�B�6�0�,�-�6�B�O�Q�[�h�s�r�h�a�f�[�O�<�5�/�+�/�<�H�U�a�b�a�U�P�H�<�<�<�<�<�<��������������������� ���������I�<�7�/�3�<�U�b�n�{ŇŗŚŔŎł�n�b�U�I�����ݽнĽ½нݽ�����"�,�+�$����ʼ����������żʼͼּؼ޼ڼּʼʼʼʼʼʾZ�X�Z�d�f�s������s�f�Z�Z�Z�Z�Z�Z�Z�Z��������������������������������������������(�)�6�6�6�)�����������������������������
����
��������������¿²©²º¿����������������������������������������������)�4�*����������������¼ʼ˼ʼ�����������������������������������������������������������	��	����"�#�%�"���������*�(� �*�6�C�H�O�V�O�C�6�*�*�*�*�*�*�*�*������������������������������������������}�s�g�Z�Y�P�[�Z�Y�Z�_�g�s�x�����������U�a�g�m�n�n�n�k�a�U�H�D�<�8�3�<�H�T�U�UÓÓÌÌÏÓÔàìñù������ùðìàÓÓ�������������������������ľʾ˾ξʾ����������������������$�0�3�3�+�$��������������s�q�i�h�s�}����������������������������������
���
�� �����������������s�n�g�g�j�s���������������������������s�������������m�c�7�6�Z�s����������������@�7�9�Y�������ֺ���!�-�*���ֺ������r�@�ֺɺƺ����ɺ�
�-�:�F�_�v�m�K�1�)���ֻ�л»��������ûлܻ���������������~�|�����������������������������������ʾǾ����������ʾ׾������������׾��z�n�n�d�n�zÇÇÈÇ�z�z�z�z�z�z�z�z�z�z�Z�O�N�F�N�Z�g�s�y�t�s�g�Z�Z�Z�Z�Z�Z�Z�Z�ù������ùϹչܹܹܹ۹Ϲùùùùùùùú���������������"�'�'�)�'�$���@�5�4�8�@�I�L�Y�c�e�r�s�~���~�r�e�Y�L�@������������������#�)�,�*�)�������E�E�E�E�E�E�E�E�E�E�E�E�FFF#F!FFE�E򹝹������������ùϹٹܹ߹ܹٹϹù�������ŭūŠŞŚşŠŭŭŹŹ��������Źŭŭŭŭ���������������������������������������ҿG�@�;�.�"�����"�.�;�<�G�J�K�G�G�G�G�߾ݾ�վѾоؾ������	������	��߻����� ����'�4�5�4�'��������������
�
�������"�*�6�:�7�6�*�"����#���������"�'�@�M�X�Y�^�Z�M�D�@�4�#�f�]�b�y���������%�%�����ּʼ�����f���{�y�p�m�l�m�y������������������������������������	���"�#�"��	���������������{�x�u�r�q�x�������������û�����������ìàÓÊÉ×ëù�����������������ì����������������������������������������EED�D�D�D�D�D�EEE*E7ECEIEMEJEAE*EE�"� ��"�.�/�;�H�T�`�V�T�H�;�4�/�"�"�"�"�������������Ŀѿݿ�����	�����ѿ�������������������������
��'�0�3�0�(�#��I�>�=�:�1�=�D�I�U�V�]�Z�V�S�I�I�I�I�I�I�������Ŀѿݿ�������������ݿڿѿĿ���čā�m�h�\�K�B�E�S�d�tāąĉđĜĥħĤčččďēĚĦĳļĿĿ��������ĿĽĳĦĚč�Y�M�@�4�'�����'�@�f�r���������r�YàßÓÏÓàìðñìàààààààààà�����������¿Ŀȿ˿Ŀ��������������������!������!�.�/�:�@�G�M�G�:�.�!�!�!�!�����#�/�2�2�/�#���������� R %  z N A 1 l E 2 / f 8 Z d = G O - C M [ X Q 3 + 0 = 7 e , o � h @ Y % R G = ) 0 $ 7 U X I a n � ( 8 r Q @ + ; �   n A 4 I D B 0 U e h _ 6    �  c  �  E  �  h  u  �  �  U  -  �  h  {  �    i  �  8  R  v  r  b  	    T  W  �  Z  �  \    �    M  �  �  N  o  `    b  �  G  8  �  K  �  1  �  �  �    �  �  k  �  �  B  �  \  �  �  z  �  �  �  %  ^  �  G  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    N  O  N  D  9  +    	  �  �  �  �  �  �  v  \  E  -  �    �  �  �  �  �  |  p  b  T  D  2      �  �  �  {  J  	  �  0  *  %               
                      z  �  }  g  K  1    �  �  �  �  �  �  �  �  v  ]  #  �  �  A  O  U  P  K  I  2  "      $  3  4  ,  "    �  �  �  O  r  a  J  5          �  �  �  �  �  c  ?    �  �  �  �    �  �  �  �  �  �  �  �  w  _  ?    �  �  �  k  .   �   �  z  u  �  �  �  �  z  k  W  >  &  �  �  �  }  R  0        �  �  �  �  �  �  �  e  5  �  �  u    �  G  �  7  �  �  �    G  t  �  �    L  n  �  �  �  }  b  8  �  �    P  �  n  �  �  �  }  t  i  Z  L  =  /      �  �  �  �  �  f  F  %  G  E  B  @  <  4  -  %        �  �  �  �  �  �  �  �  |  "  G  f  �  �  �  �  �  �  �  }  X  2  	  �  �  �  Z  D  \  �  �  �  �  �  �  �  �  �  �    '  J  m  �  �  �    '  N    )  *        �  �  �  �  �  �  ]  -  �  �  z  >    �  �  �  �  �  �  �  �  j  J  (    �  �  �  h  ;    �  �  {  �  �  �  �  �  �  p  W  ?  )    �  �  �  p  9  �  �  �  V  f  ]  S  J  A  7  ,  !         �  �  �  �  �  �  �  �  �      
    �  �  �  �  �  �  �  �  |  o  a  S  H  =  3  (  6  >  F  N  V  [  V  Q  L  G  =  0  #       �   �   �   �   �  L  F  @  :  5  /  )  "      
    �  �  �  �  �  �  �  q  �  �  �  �  �  �  x  e  Q  ;  $      �  �  �  �  [  1    �  �  �  �  �  �  �  �  �  �  y  \  8    �  �  t  @  	   �  ]  H      �  �  �  x  <  �  �  h    �  U  �  o  �  o   �  �          
     �  �  �  �  �  a  "  �  s    �  4  f  �              �  �  �  �  �  {  Z  /  �  �  0  �  5  4      �  �  �  �  �  �  n  C    �  �  C  �  �  
  �  �    �  �  
  R  �  �  �  �  �  �  �  Y    �  9  x  ~  �  �    
  �  �  �  �  �  �  �  �  �  �  x  i  Z  O  E  <  2  (  
        �  �     �  �  �  �  �  �  �  �  p  T  6  6  Y  O    �  �  �  �  �  �  �  {  T  (  �  �  �  w  :  
  �    �  �  �  �  h  4  �  �  d    �  [    �  �    ~  �  �   �  �  �  �  �  �  �  t  V  (  �  �  n  q  H    �  w  G  �  �  x  �     $  :  G  K  =  $    �  �  `    �  -  �  �  D  .  j  m  �  �  �  �  �  �    e  D  &    �  �  �  .  R  n  �  �  �  �  �  �  �  �  �  �  �  i  P  3    �  �  h  �  O    �  �  �  �  �  �  �  �  {  s  m  h  d  _  Z  i  ~  �  �  �  �  �    w  o  a  S  E  5  $      �  �  �  �  p  A     �      �    �  �    T  }  �  �  �  {  ^  8    �  �  .  �        �  �  �  �  �  �  �  �  ~  d  I  ,    �  �  �  b  �  �  �  �  �  �  w  a  I  -  	  �  �  �  w  ]  U  R  N  K  �  �  �  �  �  �  �  v  `  F  )  
  �  �  �  M    �  }  '  �  �  �  r  *  
�  
w  
!  	�  	T  �  c    �  {  �  �  �  w  �  9  b  g  [  @      �  �  |  �  �  Q  �  P  �  �    �  �  }  z  x  u  s  p  n  k  i  f  `  T  I  >  3  (        �  �  �  �  �  �  �  �  {  f  Q  <  %    �  �  �  �  _  "  �  �  �  {  n  `  N  =  ,            �  �  �  �  �  �  �  �    k  Q  @  O  J  4    �  �  �  �  �  p  J    �  �    C  0    	  �  �  �  �  �  �  s  9  �  �  �  c  B  "     �    }  |  {  t  l  d  Y  L  ?  1  !      �  �  �  �  F                   �  �  �  �  �  ^  -  �  �  K  �  b   �  %      �  �  �  �  �  b  1  �  �  e  E    �  q  	  �  4  	    �  �  �  �  �  �  �  �  �  �  �  k  T  >     �   �   �  �  �  �  �  �  �  {  q  g  Z  N  A  3  #      �  �  �  �  �  �  �  �  �  �  q  M  !  �  �  {  L    �  �  q  *    �    '      �  �  �  M  �  �  B  �  �  ;  �  �  L  h  x    �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  m  $  �  t    c  1  �  
  !  "  �  �  t    �    `  k  [  �  �  	�  u  �  ,      �  �  �  �  �  �  �  �  d  C  #    �  �  �  n  K  �  �  �  �  �  �  �  �  �  d  :    �  q    �  (  �    �  l  �  �  �  �  �  �  �  �  {  T  &  �  �  Z    �  �  u    �  �  �  �  �  �  �  �  s  Z  B  )    �  �  �  �  Z  -     �  �  �  �  x  f  S  @  .      �  �  �  �  �  {  Z  6      -  ,  $    �  �  �  �  Q    �  r  �  {  �  W  �  E    �  t  Y  >  !    �  �  �  �  {  [  3  �  �  h    �  �  r  �  �  �  �  b  =    �  �  �  K    �  �  V  �  �    A   �  �  �  �  �  ~  r  g  [  Q  K  F  @  E  R  `  m  ]  E  -    �  �  �  �  �  �  {  v  p  k  e  `  Z  R  C  5  '    
  �  �  �  �  �  �  �  �  |  u  a  D  %    �  �  �  i  3  �  l  g  d  `  \  V  O  H  9  &    �  �  �  �  �  p  M     �  �