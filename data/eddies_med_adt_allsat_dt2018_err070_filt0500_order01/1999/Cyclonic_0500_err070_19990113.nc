CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�Q��R      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�q�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ȴ9   max       <�C�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>z�G�{   max       @F�Q��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �θQ�     max       @vb=p��
     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�j           �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �"��   max       <#�
      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�r�   max       B2�      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�jv   max       B1L�      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >{�X   max       C���      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >XU�   max       C���      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          _      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��K      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��X�e   max       ?�X�e,      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���`   max       <�C�      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @F�\(�     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �θQ�     max       @va�����     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @��          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�u%F
�   max       ?��)^�     `  U�               
         _      	               #   -      	         6         1   '      	      -                     !                        	   ,   /   9               $   (            (         /   	   >Nn�N/n�N1IN��N�;�O��EO���P�q�O�M0N���O�O�n�Np>N�ŹP�7P�O	7Oj�N�8N��sO�6DO���O5��P{Pn�N��Ni�5O��7O��O�l�OB��N7��N���OQ�O�dO�ihN�B�O���M��O��NC�Ná~M�mN��6Oމ�P9qqO�_\N�q�O�X�N�sKN.%_O�ZP
,Ou��N��9Ns��O�P3O(-�O(�O}]QN�cUO��D<�C�<t�;�`B;�`B;D��;D��;o:�o%   ���
�ě��ě��49X�49X�49X�u�u���㼣�
���
��9X�ě����ͼ��ͼ�����/��h������w��w�#�
�'',1�,1�,1�8Q�8Q�D���D���D���L�ͽP�`�T���Y��Y��aG��aG��aG��ixսixսq���y�#��%��o�����7L���T���T��{�ȴ9�����������������������������������������������������������������������������������������������������������������������������������������t}���#U{������nU<#�������������������||}���	����������V[agt����|trg[YUTVV')6BIV]nv���thOB6)&'inwz����zniiiiiiiiii��������������������(,6C\hu������uh\6*&(P[mz�������naTH=202P #<HU_abaUSIH<3/$#! ��������������������(/<HQSHG<5/-((((((((

#*/020&#
	

'/BO[ht����tnbOB<;-'$/<HU\^[USTZUH<#$�����������}�����������������������������!+!������������


����������)5951-)����	";GJH;"	�����������������������������

�������������������������������������������������������������������������������������������wz���������zxtttwwwwNN[gt���������tg[PNN���������������������������������������������������������#0<FUZchhbUM<0#"
#,0310#y{������������}{xtsy����������������������������������������NVgt����������tg[RON�2Bgw��yk[N5 ���)3;?=7)������������������������R[t����������tg[UPPR������	

��������KNU[gige^[NMKKKKKKKK<HUfsz�����znaUH<::<rt{����������������r���������������������������������������������������~�����������!.GHB5�����,03<FITU]][UH<30%$%,Y[^gt��������|tge[XY
#(-<H]UM</#
 � enz~�������znhbeeee)5B[gmgbcb^[JB5)�$������$�0�3�8�0�.�$�$�$�$�$�$�$�$�H�<�<�:�<�C�H�N�U�V�V�U�H�H�H�H�H�H�H�H����������
�����
� ���������������������������������������������������޾4�0�(�$�!�(�4�A�E�J�F�A�4�4�4�4�4�4�4�4�`�G�;�.�#�*�5�G�T�m�y�������������y�m�`���޾׾ʾʾ׾����	�"�(�6�<�;�"��������������]�B�5�7�G�s������������������ݿοɿſѿݿ��������"�$������ݾ�x�t�~��������������������������������~�������������������Ŀÿ����������;�1���"�G�T�`�m�t�y�������������y�m�;�U�P�H�G�F�H�U�X�Z�X�U�U�U�U�U�U�U�U�U�U�H�G�C�G�H�K�U�a�g�n�o�r�n�a�a�U�H�H�H�H��ʾ������������������ʿ�����	����������ٿ�����(�5�Z�x�������s�Z�N��)�"�$�#�"�)�4�6�=�@�B�O�O�W�O�K�B�?�6�)�M�A�;�A�G�M�Z�f�s�������~�u�s�f�c�Z�M�
� �����
��"�#�(�#���
�
�
�
�
�
�
�
�����������������ɺͺֺٺֺѺɺ����������l�_�P�B�8�6�:�F�S�_�x�������»��������l�M�9�1�2�8�A�M�Z�f�s�������������s�f�M����(�4�9�A�F�K�F�B�A�4�(���	��������ۻлܻ����'�@�I�I�D�5�*�����ʼ����ż��ʼּ����.�@�C�B�<�!���ּ�ììåììøù����������������ùìììì¿³²©¬²µ¿��������¿¿¿¿¿¿¿¿��������������������� ����&�*�(��	��������Źż������������*�/�*�������߿��������Ϳѿݿ�����#�������ѿĿ��b�Z�N�A�=�7�A�H�N�Z�g�s�������������s�b�ܹ׹ܹ߹����	�������ּܹܹܹܹܹܹܹܼϼԼּ�������������ּּּּּ��Z�N�N�I�N�V�Z�g�s���������������s�g�Z�Z����������������$�&� �������������������������	���"�&�*�+�*�#��	����������ھ׾оϾ׾��������� ����������-�6�C�O�W�qƀ�~�u�O�C�6�*���r�o�e�Y�S�Y�d�e�r�t�r�o�r�t�r�r�r�r�r�r�������������Ľнݽ���������н������������������������������������������'�4�>�@�M�Y�\�Y�M�M�S�M�I�@�4�'�����������������������������������������ŠŚŔŇŀŇŔŝŠŭŹſ��������ŹŭŠŠ�V�=�#������$�0�=�V�b�o�|�|�z�o�b�V���v�t������������������������������������������(�5�J�Z�_�d�Z�N�5�(�����S�S�S�W�_�d�l�v�x���������x�l�_�S�S�S�S������������������������)�B�?�8�)��������������������������������������������ŔōŇł�~ŇŏŔŠŢŢŠŔŔŔŔŔŔŔŔ�������������ùйܹ���������۹�������Ěčā�Z�D�6�B�hčĚĳĿ������������ĿĚ�ɺ����������������ɺֺں���������ɺ3�0�'�&�'�-�3�@�G�L�Y�e�o�e�Y�N�L�@�<�3�������������������������������������������	�	���)�O�tāćċĄ��v�h�O�;�*��������������ûŻлܻ��������лû����#��������#�0�9�<�G�I�K�L�I�:�0�#ECE8E7EGEPE\EuE~E�E�E�E�E�E�E�E�EuEiE\EC�������������������ĿοѿҿֿѿĿ�������Å�_�L�@�A�D�H�P�nÇÓìñùÿ����ìàÅ 7 \ E   ;   ) Q < ? 6 O  L A t S b 6 H > 0 4 1 o 3 T P 1 S : O Y . 4 O e Q c W 7 ] � P * R 6 8 r H P U e < u V g ;  g H G  �  k  E  �  �    �  �    �  5    p    �  ~  p  e  �  �  �  &  �  Y    �  z  �  �  �  �  i  �  :  J  9  Y  s  2  �  &    �  �    ]  �  �  �    O  �    �    s  �  s  c  j  �  T<#�
:�o%   �49X��`B�ě���o��vɼ�C��T�����
���ͼ�C����8Q�u�C���/��/������L�ͽ49X���P��o��w��㽅����O߽y�#�49X�ixս]/��%���P�e`B���P�@���t��L�ͽY��T���u�\������G����-��{�y�#�y�#��vɽȴ9���w��O߽�\)������������J��vɾ"��BĩB ��B6tB�bBQ+B+�B /�B&�DB*�B�B	Z�B�mBH�Bw�B2�A��B�nB E�BǅB$�6B4�B��BU�B�B-�2B߼B� A�r�B]NB�7B)�B!dB�4B�_B OHB	��B��B��B :�B&QB%[�B)kGB��B��B
UB��B*�B ~B
�B�*B��B�KB	B��B�B
�B�B&Z�B	�nB��BG�B��B�NB �BA�B$VBDB+8�B ��B'��B)�PB]B	?^B@�B>�B��B1L�A�~�B�B >-B�XB%�B=�B��B�pB?�B.9B�OB�lA�jvBBB� B-hB!p�BL�Bg�B =]B	� B�:B5�B B�B&+MB%=�B)KBGB��B	�B3B>�B ��B	�6B9�B�.B�/B�\B�TB�B
�B;�B&�7B	AjBI�B��B?$B	��AĔ�A�!�A���A9L�AiHAZ��A�z�A�d4AG��AsA�Ahl�A�k�Aś2AR�(A�A�DA@B,A��@*(�@�Y�A?|�A6�-@��jA��A��A�z�A���A��A|�sA��?�A�A�G�AӁ�A[@BAV?�B �?�_�A&��A1'@ϦA��A�rBCA���A��T@���A��A�8fA�J�>{�XAޭ�@7S�?��gB�A��r@�47A�0aC���Aw��A�"DB	��A�A�a9A���A9[Ah��AY9DA��A�UyAF�*Ar�\Ak<AŅ�A��AR҃A��A׎4A@mDA��@.�z@��A>�XA6�@��~A	%A͹*A�� A��A��ZA~��A���?A�wA�^A�A[!AU��B ��?�,�A##A1��@��A�
�A��mB!>A��TA�v[@�<�A�UA��.A�}F>XU�A���@;y?���B>�Aڇ[@��DA�MC���Ax��Aɛ�      	                  _      	               #   .      	         7         2   '      
      -                     !                        
   ,   /   :               %   (            (   	      /   	   >                     #   C            %         )   /               '         %   1               !                              #               #   /         '         !   /            '               #                        C            %         '                              %                                             #               !   /         #         !   )            %               !Nn�N/n�N1IN�>�N�;�OB�"O�bP��KO/��Nd��N��O�n�Np>N���P�SO�;�O	7Oj�N�8N��sO/�OJׇO5��O�i{O�Z_N��Ni�5O
��Ov�O���N�n�N7��N���N�uAO�dO^\N�^O���M��O�opNC�N��M�mN��6O�J�P9qqO��N�q�Oû�N�sKN.%_O�ZO���Ou��N��9Ns��O�FuO(-�O(�O>�0N�cUO���  �  �  E  c  2  6  �  �  '  �  �    �  �  0  �  P  �  ]    �  �  �  �  `    J  ~  n  �  �  �  �  �  �  k  �  t  �  #  Z  �  �  (  8  ?  	�    t  �  6  |  �  �  �  ,  �  x    �  �  
<�C�<t�;�`B;�o;D���ě�%   �D����o��`B��`B�ě��49X�D���u��`B�u���㼣�
���
�8Q��h���ͽ'���/��h�D���D���,1�49X�#�
�'49X�,1�<j�0 Ž8Q�8Q�L�ͽD���H�9�L�ͽP�`�]/�Y��m�h�aG��e`B�aG��ixսixս}�y�#��%��o��+��7L���T��9X��{���`������������������������������������������������������������������������������������������������������������������������}�����������������y}�#U{������{U<#�������������������������������������W[dgrt���~{tpg[YUTWW')6BIV]nv���thOB6)&'inwz����zniiiiiiiiii��������������������*/;O\hu�������uhC0**JWamz��xmf`THB><?J #<HU_abaUSIH<3/$#! ��������������������(/<HQSHG<5/-((((((((

#*/020&#
	

<BCLO[hltuusnh[OBA<<"'-/<HUXZXQNH</$# "�����������}������������������������������� &($������������


����������)5951-)�����	""$"	������������������������������
�������������������������������������������������������������������������������������������wz���������zxtttwwwwQ[gt���������tg^[SPQ���������������������������������������������������������#0<ISXbggbUPA<0#,0310#w{�������������{yvww����������������������������������������Wgt����������tg[SPOW�2Bgw��yk[N5 ����)09=;5) �������������������������S[t����������tg[VQPS������	

��������KNU[gige^[NMKKKKKKKK<HUfsz�����znaUH<::<�����������������uw����������������������������������������������������~�����������!.GGB5�����,03<FITU]][UH<30%$%,Y[^gt��������|tge[XY
#&*/9<6/#
 enz~�������znhbeeee)5BN_b^aa\[HB5)�$������$�0�3�8�0�.�$�$�$�$�$�$�$�$�H�<�<�:�<�C�H�N�U�V�V�U�H�H�H�H�H�H�H�H����������
�����
� ���������������������������������������������������޾4�0�(�$�!�(�4�A�E�J�F�A�4�4�4�4�4�4�4�4�T�O�G�>�8�:�G�T�`�m�y�}���������y�m�`�T�	������о׾����	�"�%�3�9�5�.�"��	�������`�E�=�@�O�s�������������������ؿ�ݿֿѿппѿݿ����
�����������~�x����������������������������������������������������¿������������;�1���"�G�T�`�m�t�y�������������y�m�;�U�P�H�G�F�H�U�X�Z�X�U�U�U�U�U�U�U�U�U�U�H�H�D�H�H�K�U�a�f�n�n�q�n�a�_�U�H�H�H�H��ʾ������������������׾������	�����5��������(�5�A�Q�Z�g�z�z�s�g�Z�N�5�)�"�$�#�"�)�4�6�=�@�B�O�O�W�O�K�B�?�6�)�M�A�;�A�G�M�Z�f�s�������~�u�s�f�c�Z�M�
� �����
��"�#�(�#���
�
�
�
�
�
�
�
�����������������ɺͺֺٺֺѺɺ����������l�b�_�S�P�E�C�F�Q�S�_�l�x�����������x�l�Z�M�A�=�4�5�=�A�M�Z�f�s������~�s�f�Z����(�4�9�A�F�K�F�B�A�4�(���	���������������'�4�>�C�@�;�4�'����ּϼɼڼ������!�.�=�A�?�:�"������ììåììøù����������������ùìììì¿³²©¬²µ¿��������¿¿¿¿¿¿¿¿�	��������������������	���� ���	�	�������������������������������뿫�������Ŀѿݿ�����������ݿѿĿ��Z�N�N�O�Z�g�s�����������s�g�Z�Z�Z�Z�Z�Z�ܹ׹ܹ߹����	�������ּܹܹܹܹܹܹܹܼϼԼּ�������������ּּּּּ��Z�T�Q�Z�`�g�s���������������s�g�Z�Z�Z�Z����������������$�&� ������������������������	���"�#�(�)�'�"���	������۾׾Ѿо׾������ ���������������-�6�C�O�W�qƀ�~�u�O�C�6�*���r�o�e�Y�S�Y�d�e�r�t�r�o�r�t�r�r�r�r�r�r�������������Ľнݽ���������۽Ľ���������������������������������������'� ��'�4�5�@�@�M�N�M�I�M�R�M�D�@�4�'�'����������������������������������������ŠŚŔŇŀŇŔŝŠŭŹſ��������ŹŭŠŠ�=�&�����0�=�V�b�i�o�u�{�{�y�o�b�V�=���v�t�������������������������������������������(�4�G�N�T�Z�H�5�(����S�S�S�W�_�d�l�v�x���������x�l�_�S�S�S�S������������������������)�@�>�6�)��������������������������������������������ŔōŇł�~ŇŏŔŠŢŢŠŔŔŔŔŔŔŔŔ�������������ùйܹ���������۹��������t�[�F�7�B�O�hčĚĳ����������ľĦĚč�t�ɺ����������������ɺֺں���������ɺ3�0�'�&�'�-�3�@�G�L�Y�e�o�e�Y�N�L�@�<�3�������������������������������������������
�
���)�O�tāĆĊă�~�u�h�O�9�)��������������ûŻлܻ��������лû����#��������#�0�9�<�G�I�K�L�I�:�0�#ECE=E;ECEJEPE\EfEuE�E�E�E�E�E}EuEiE\EPEC�������������������ĿοѿҿֿѿĿ�������ÓÇ�b�O�C�D�H�U�n�zÓìîùý����ìàÓ 7 \ E  ; * + O 5 H 0 O  H @ k S b 6 H 1 5 4 A V 3 T 8 % G - O Y ( 4 I Q Q c ] 7 U � P * R + 8 n H P U _ < u V g ;  L H B  �  k  E  �  �  �  W  �  u  }      p  �  q  �  p  e  �  �  }  �  �  a    �  z  8  �  )    i  �  �  J  �    s  2  �  &  �  �  �  �  ]  Z  �  a    O  �  �  �    s  �  s  c  �  �  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  �  �  �  �  �  �  �  z  f  R  <  '    �  �  �  �  �  �  �  �  y  o  b  N  :  $    �  �  �  �  {  V  0    �  �  E  I  M  Q  T  W  Y  Z  [  ]  _  `  _  ]  Z  T  M  A  2  $  ,  G  `  c  Y  K  :  (    �  �  �  �  �  �  p  Z  :  �  �  2  /  ,  $    	  �  �  �  �  �  i  @    �  �  �  �  �  �  �  �      (  1  5  6  /  $      �  �  �  ~  N    �  �  �  �  �  �  �  �  �  �    p  _  O  >  ,      �  �  �  V  �  �  �  �  �  _  8    �  �  {  C  �  �  5  �  g  �    �        #  &         �  �  �  �  �  �  h  <    �  �  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  _  C    �  �  �  3  �  v    	     �  �  �  �  �  �    f  N  7  D    �  �  f  b    �  �        #  ,  /  -  ,  '              $  .  9  �  �  p  A  j  M  +    �  �  G  �  �  9  �  p    �  C   �    *  /  -  (  (      �  �  �  �  S    �  p    �  [   �  *       -  �  �  �  �  �  �  m  e  e  Q    �  P  �  �  a  P  B  1      �  �  �  �  �  �  O    �  p    �  �  ]  '  �  �  �  �  �  �  u  i  X  G  2      �  �  �  �  �  �  i  ]  Y  V  R  L  G  B  <  6  .  &            �  �  �  �        	  �  �  �  �  �  �  �  �  �  u  b  M  7  "    �  �    $  >  �  �  �  �  �  �  �  �  F  �  y  �  Z  ~        Y  w  �  �  �  �  �  �  `  3  �  �  s  #  �  M  �  �  �  �  �  �  �  �  �  q  ]  F  +    �  �  �  �  j  D    �  �  u  �    @  `  w  �  �  �  t  N    �  z    �    m  �  
   �  �  N  `  W  C  4  #    �  �  �  W    �  �  6  �  1  �  )            	     �  �  �  �  �  �  Y  1  
  �  �  �  @  J  H  G  J  N  P  P  P  N  J  D  ;  -       �  �    �  C  z  r  �  �  	  ;  d  }  z  l  T  5    �  �  c  �  ~    �    ;  U  h  n  k  ^  K  2    �  �  H  �  �  /  �  �  '  \  �  �  �  �  �  �  �  �  �  ^  .  �  �  m    �  X  �  ~  �  �  �  }  �  �  �  �  r  ]  E  ,    �  �  �  P    �    �  �  s  `  M  :  )           �  �  �  �  �  �  �  �  �  �  �  �  }  ^  i  �  a  4    �  �  `  '  �  �  z  O    �  �  P  d  t  �  �  �  �  t  b  M  7    �  �  �  �  {  [    �  �  �  �  �  �  �  �  �  x  V  .    �  �  ;  �  x  �  A  �  b  h  k  i  b  W  J  7  !    �  �  i    �  M  �    u  �  c  �  �  �  �  |  j  W  B  )    �  �  s  9  �  �  �  K    t  a  [  N  <  "    �  �  �  W    �  �  8  �  �  �  ]  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  u  n  g  _  X  �    "      �  �  �  s  �  m  W  F  :    �  �  Z    �  Z  S  M  F  ?  8  2  +  $      
      �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  ~  l  Z  H  7  %  
   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  (         �  �  �  �  �  �  p  S  3    �  �  �  h      �  )  8  *    �  �  �  n  ;  �  �  [  �  y  �  8  �  +  �  �  ?  :  $    �  �  �  m  7  �  �  h  #    �  �  N  �  #  �  	�  	�  	�  	�  	�  	�  	{  	S  	  �      �  !  �  �  E  k  �   �    �  �  �  k  K  +    �  �  �  �  �  <  �  �  9  �  �  e  V  q  b  D            �  �  �  f  0  �  �  d    �  I   �  �  t  a  M  3    �  �  �  �  �  �  k  P  :  $     �   �   �  6  1  ,  '  !      
    �  �  �  �  �  `  '  �  �  b  !  |  p  e  r  z  n  U  ,    �  �  c    �  �  U    �  %  J  �  �  �  �  �  �  t  R  8  �  �  4  �  :  �  ~  �  �  U  F  �  �  �  �  �  �  ~  b  A    �  �  �  e  =    �  �  �  O  �  �  �  �  p  Y  A  A  M  Y  V  D  3       �  �  �  �  p  ,    �  �  �  �  o  K  $  �  �  �  r  J  4       �   �   �  �  �  w  f  O  -  �  �  �  J    �  r  +  �  [  �  a  r  Q  x  j  \  N  A  4  )        �  �  �  �  �  �  {  ^  E  ,      
  �  �  �  �  �  �  �  �  �  e  7  �  �  �  <  �  �  
�  
�  d  �  �  X  /  
�  
�  
�  
-  	�  	F  �  �     6  .  �  �  �  �  �  ~  o  _  Q  B  2  !    �  �  �  �  �  �  �  �  X  	�  
  
  	�  	�  	�  	X  	  �  {  5  �  �  `  �  n  �  L  �  �