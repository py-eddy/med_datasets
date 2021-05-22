CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��vȴ9X      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N �L   max       P�s�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��l�   max       =0 �      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @F��\)     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @v���
=p     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @O@           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @��`          �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �n�   max       =�P      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�E�   max       B4�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4Ñ      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >� ?   max       C��      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�k�   max       C��9      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N �L   max       P�s�      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�j   max       ?��6��      �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��l�   max       =0 �      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @F��\)     	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v�          	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @O@           |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @���          �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�0��(�   max       ?��6��       T�                              %            /   /   /      
      ,            9          )   &   #                     >      
         $   
   *                  =         
                  
         N.�N�9'N��NO��O���OKx�NtAO��OOhEO-�AO!\2N3j�O߲�OIƠP�s�P�O��O@8iO�:�P��N �LO*�ZN�`bPy0<NEnkO~� Oa�O�*xO�7�Nm<5O(��Nn(�O�(�N�A N��MP�O�5RN���N���OS��P��N�G�O�i�N�A2N��N��O2#�N �P��N�6�O'�N��uN;-�N���N�B�NC��N���Nc��Nc0NE&�N���=0 �<t�;�o:�o:�o��o�D�����
�o�#�
�#�
�#�
�e`B��o��o��9X�ě�������/��/��`B��`B��`B��h�C��C��C��\)��P��P��P��P��P����w��w�#�
�',1�49X�49X�8Q�@��@��D���D���H�9�P�`�aG��e`B�q���y�#��%�����+��+��7L��{��{�\��l�GHUX`ZUMH=GGGGGGGGGG16;BDOOXOJB63,111111��������������������".*$"�������������������������������������;BOQ[][OB>;;;;;;;;;;����������������������������O[hirt}���}th[OKIIKO��������������������).5653)#8Bgt���������xtcWFB8!#'/<HO\addeaSH</*#!�
#CIUn{����{\<���LUanz������zna^WTLCL26BOWhvvtmh[OKC62*12FOQSY[htwy�����th[OFamz������������mc^\a����6BKMG6���������������������������������������������uz������������zuuuu'0;>2,+5hrz|���xB6' 
         ��
#/<FDHPNH/#
���������������������
0;IR[`hnwsmh[6
�������)6A=6)%�������������}||����������������������������������������������������������������������������������������������������������=BHUanz�����zaUHD=;=KYgt������������tgNK	
!#,,$#

				��������������������Y\gt�����������tga\Y����������������������������������������;HUan|����maUHC>?<:;#/05:810'#�������������������������������������������������������������������������)BNZSLHLHB5)	TUZanzz|znlcaUTSTTTT�������������������������� ���������������������������������/039<DINSRPKIA<40.///07<HIUZ\[YWULIHD<0/lno{��~{nllllllllll45BFN[agnhg[WNBA5344������������������������������������������������������������HNNKH<5115<HHHHHHHHHŭŬŬŭŹ��������Źŭŭŭŭŭŭŭŭŭŭ�#�����"�#�/�6�<�B�@�<�/�#�#�#�#�#�#D�D�D�D�D�D�D�D�D�EEEEEED�D�D�D�D��	��������	���"�&�"��	�	�	�	�	�	�	�	àÚÐÇ�}�m�n�zÇÓàìù��������ùìà�U�Q�H�9�3�8�<�H�U�a�n�v�}ÇËÇ�z�n�a�U�a�Z�_�a�l�n�r�z�r�n�a�a�a�a�a�a�a�a�a�a�{�r�n�j�f�f�n�v�{ŇŔŠŢšŠřŔŇ�{�{���
�����#�,�/�<�@�A�C�A�=�7�/�#��L�I�K�L�Y�Z�e�m�r�v�~���������~�r�e�Y�L�m�d�`�T�L�I�T�`�m�m�y�}�������������y�m�s�l�p�s�~�������������|�s�s�s�s�s�s�s�s���������z�������Ŀѿ�������ѿĿ������M�H�A�;�9�=�A�M�`�f�s�����{�s�l�f�Z�M�������z�Z�P�D�4�5�A�s�������������������N�D�@�D�Q�g�������������������������g�N�T�J�H�L�T�`�y�������ÿɿǿĿ����y�m�`�T�нɽĽ��������������ýнݽ������ս���ļĺ�����������0�F�J�I�<�#���������˼ʼüӼؼ���!�.�:�B�G�I�G�:��������Z�Z�W�Z�`�g�h�i�q�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�.�)�&�"�)�.�;�G�T�U�`�e�k�`�^�T�G�;�.�.�����������������������������������������T�H�/�	�������s�z����������"�;�m�o�m�T�׾ϾʾȾʾ׾�������׾׾׾׾׾׾׾׿ѿĿ����������������¿Ŀѿҿ����ݿ������	���5�A�N�Q�Y�S�N�J�<�5�(����ֹܹϹù������������ùϹܹ��������������л������������������������ûܻ������������������������¾��������������������������������������ʾ̾ھؾ׾оʾ��������ݿѿϿѿݿݿ������� �������m�`�X�O�L�T�`�m�����������ɿĿ������y�m��� �����$�0�4�5�1�0�$�$�������������ƻ������������������������������F$FFFFFFF$F=F^FoF�F�F�FFnFcFJF=F$�����y�p�j�i�U�[�z���������������������������������������������������������������������ھ��������	�
�	���������������߾߾��������	����!����	���𺰺��~�r�z�������ֺ���!�-�E�-����ֺ��������������$�)�*�)���������z�O�8�/�1�H�a�z�����������������������z�����������������ûлѻܻ߻ܻٻлû������5�3�)�(�)�5�B�N�N�O�Q�N�C�B�5�5�5�5�5�5�Y�O�M�@�@�@�E�M�Y�f�j�m�g�f�Y�Y�Y�Y�Y�Y�6�2�1�/�0�6�C�O�S�\�d�u�zƀ�u�h�\�O�C�6�����������������������������������������U�H�@�;�9�<�H�nÇàù����������ìÎ�n�U�	�����������	�
�
��"�$�&�"��	�	�	�	�#�!�#�������#�0�8�:�=�<�:�5�0�&�#�ɺ��������ºɺҺֺ����ֺɺɺɺɺɺ�ìçà×àìù��������ùìììììììì�(�$����������(�4�9�A�E�A�;�4�(�(�����������������������ĽɽнܽнĽ������'�����'�4�@�C�D�@�4�'�'�'�'�'�'�'�'�Ľ½������������Ľнѽ۽ݽ�޽ݽӽнĽ�������������������������������������������$�%�$�#� ���������ĳıĭīįĳĿ��ĿĿĿ����ĿĳĳĳĳĳĳD�D�D�D�D�D�D�D�D�D�D~D�D�D�D�D�D�D�D�D� H + : u R < X + V " = g P L X W @ > g X [  ) u E 9 - f l 5   B F ; C $ l 2 w Q W M D 5 \ ' A F [ U l 6 D P I G J @ K } ;    /  �  �  f  o  �  U  !  �  p  e  �  3  �  �  +  �  �  i  *  =  i  �    Z    �  �  R  {  m  r  Y    �  h  n  �  �  �    �    �  ^  �  �  0  �  �  p  �  `  �    M    y  �  �  �=�P;ě��e`B�o��/�ě��49X�ě���`B�<j���
�e`B�#�
�}󶽁%��C��<j�\)�e`B��\)���T���C���{�#�
���������t���hs�,1�u�'aG��D���,1����}�L�ͽ<j��o���
�aG���E��u�L�ͽ�%�]/�aG�����%��t���\)���-��-���
��hs���P�\��vɽ���n�B�B0�B`�A�E�BWB!5�B��BxhB��B�NB�	By�B	��B��B&N�BqbB�#B�B \�B.k1B�,Br]B�0BJB�B�BIBV�B[�BB!��B)��B+Z�B��Ba�B�zB
��B$�B�B
S�B�B��B9�B%JPB
��B!d�B��B4�B�8BuBQpB#!�B4�B&s�B&��B(��BEiBi(B�B��B�jB��B �B�A���B�HB �B��B@6By�BM�B�uB��B	{�B�=B&�\B�BǏB3A��tB.�$BJ!BB�B?�B�iB�YBG�B�B$�BQWBƊB!�B)�QB+1ZB�	B?�B0�BaB$�NB}�B
A�B��B��B?�B%D�B
��B!@B��B4ÑB��B?EB��B#8�B>�B&@�B&��B(�PB96B��B8�BwB��A���A�ENC�@A��Aˣ�A���A�!�A�.6A�	{?��Aj�A�EbAw%XA>�	A�F_A�Z�Ao?)A(ƕA�\A	N�A�_�Ad}�A���A�IeAT!]Ay�
A��p>� ?@��AKՌAM��AAo�B	d�B��C��A�S�@�)AW}�AYr�@>7�A��=A�-�@��sA�%T@ٝB]�AJ�A�+mA[�A�	�@7��A���A5�(A#߻@�E�A'��Bd�B	�A�w�C��'A���A�C�8	A�Â�Aĭ�A�hA��A���?��!Ai��A���Au~A>ɛA��LA�o�Am��A)>�A�!�A�A�zqAf/�A��nA�F
ASK�Az��A�}�>�k�@��AKFAM�A�sAm��B	�B��C��9A��e@�eAV�AZCu@C��A�~9A���@�cA�{�@���Bl)AJ�Aǜ�A[A�^�@4��À�A7
�A"�K@���A(��BBsB	?<A�o�C���                     	         %            /   0   /      
      ,            9          *   &   #                     ?      
         %      +                  =         
                     	                                             %      =   -   !      )   -            ?            #   )                     %   )            -      '                  '                                                                           '      =   +   !         -            =                                    #   )            -      !                                                      N.�N�9'N��NO��O8�XN�-�NtAO��OOhEO'�O!\2N3j�O�Y�O5P�s�PY�O���O@8iO��KP��N �LO�yN�`bPq��NEnkO)lCOa�O]kuO�&qNm<5N��FNn(�Os�rN�A N��MP kO�5RN���N���O(W^P��N�G�O��FN�>�N��N��O2#�N �OZadN�6�N��kN��uN;-�N���N�B�NC��N���Nc��Nc0NE&�N���  ^  �  !  C  �  �    �  M  Z  1  �  �  
  1  
  �  �    G  �  :  �  �    �  4  �  R  �    W  3  c  q  �  6  �  "  �     �  �  �  �  *  ]  �  z    �  �  �  �  +  h  �  �      =0 �<t�;�o:�o�ě��ě��D�����
�o�49X�#�
�#�
��o�ě���o�ě����ͼ����+��/��`B����`B���C��'C��0 Ž'�P�'�P��w����w�0 Ž#�
�',1�@��49X�8Q�Y��D���D���D���H�9�P�`�����e`B�u�y�#��%�����+��+��7L��{��{�\��l�GHUX`ZUMH=GGGGGGGGGG16;BDOOXOJB63,111111��������������������".*$"��������������������������������������;BOQ[][OB>;;;;;;;;;;����������������������������KOR[hrt|��|th[QOKIIK��������������������).5653)#>BNgt��������{teXHB>+/<HJUX]^XUHC</.&$++�
#CIUn{����{\<���ENUanz������wngYVUME36BOguutnlh[MD753,43FOQSY[htwy�����th[OF`agmz�����������zga`����6BKMG6����������������������������������������������uz������������zuuuu;?9,*,6hqz|���w]B)$; 
         
"#/;<@><4/#

��������������������)6BO[hiroif[O6"������#"������������������}||����������������������������������������������������������������������������������������������������������HUaiz������zaUHA<=?HKYgt������������tgNK	
!#,,$#

				��������������������^git�����������tgd_^����������������������������������������IUanx~����rdaUH?>?AI#+027600&#�������������������������������������������������������������������������)5@BGFD@65)&TUZanzz|znlcaUTSTTTT�������������������������� ���������������������������������/039<DINSRPKIA<40.///07<HIUZ\[YWULIHD<0/lno{��~{nllllllllll45BFN[agnhg[WNBA5344������������������������������������������������������������HNNKH<5115<HHHHHHHHHŭŬŬŭŹ��������Źŭŭŭŭŭŭŭŭŭŭ�#�����"�#�/�6�<�B�@�<�/�#�#�#�#�#�#D�D�D�D�D�D�D�D�D�EEEEEED�D�D�D�D��	��������	���"�&�"��	�	�	�	�	�	�	�	ìàÔÏÉÅÆÑÓàìòù��������úùì�H�F�<�<�7�<�H�U�_�a�n�q�v�q�n�h�a�a�U�H�a�Z�_�a�l�n�r�z�r�n�a�a�a�a�a�a�a�a�a�a�{�r�n�j�f�f�n�v�{ŇŔŠŢšŠřŔŇ�{�{���
�����#�,�/�<�@�A�C�A�=�7�/�#��Y�N�L�J�L�Y�[�e�r�u�~�����������~�r�e�Y�m�d�`�T�L�I�T�`�m�m�y�}�������������y�m�s�l�p�s�~�������������|�s�s�s�s�s�s�s�s�����������������Ŀѿ�������ѿĿ������A�>�<�@�A�J�M�Z�f�s�v�v�s�o�h�f�Z�M�A�A�������z�Z�P�D�4�5�A�s�������������������g�N�F�B�F�R�g�������������������������g�T�M�K�O�`�y�����������ȿĿ������y�m�`�T�нɽĽ��������������ýнݽ������ս����������������������
�#�.�#�!��������ʼüӼؼ���!�.�:�B�G�I�G�:��������Z�Z�W�Z�`�g�h�i�q�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�.�,�(�%�+�.�;�G�T�T�`�d�i�`�[�T�G�;�.�.�����������������������������������������/�	���������t�z����������"�;�T�k�m�a�/�׾ϾʾȾʾ׾�������׾׾׾׾׾׾׾׿��������������Ŀѿӿݿ�����ݿѿĿ������	���5�A�N�Q�Y�S�N�J�<�5�(���ܹڹ׹ҹù��������ùϹܹ���������ܻлû����������������ûͻ����������ܻо��������������������¾����������������������������������������Ⱦʾ־Ӿʾƾ��������ݿѿϿѿݿݿ������� �������y�m�`�Z�Q�O�T�`�m���������������������y��� �����$�0�4�5�1�0�$�$�������������ƻ������������������������������FFFFFF$F1F=FJFXF|F�F�F}FkFVF=F1F$F�����y�p�j�i�U�[�z���������������������������������������������������������������������ھ��������	�
�	������������������������	�
�������	���𺰺��~�r�z�������ֺ���!�-�E�-����ֺ��������������$�)�*�)���������T�=�5�7�H�a�m�z�������������������z�m�T�������������������ûллܻ޻ܻ׻лû����5�3�)�(�)�5�B�N�N�O�Q�N�C�B�5�5�5�5�5�5�Y�O�M�@�@�@�E�M�Y�f�j�m�g�f�Y�Y�Y�Y�Y�Y�6�2�1�/�0�6�C�O�S�\�d�u�zƀ�u�h�\�O�C�6�����������������������������������������n�a�U�N�K�N�U�W�a�n�zÇàâåàÖÇ�z�n�	�����������	�
�
��"�$�&�"��	�	�	�	�0�'�#�������#�0�7�9�<�8�3�0�0�0�0�ɺ��������ºɺҺֺ����ֺɺɺɺɺɺ�ìçà×àìù��������ùìììììììì�(�$����������(�4�9�A�E�A�;�4�(�(�����������������������ĽɽнܽнĽ������'�����'�4�@�C�D�@�4�'�'�'�'�'�'�'�'�Ľ½������������Ľнѽ۽ݽ�޽ݽӽнĽ�������������������������������������������$�%�$�#� ���������ĳıĭīįĳĿ��ĿĿĿ����ĿĳĳĳĳĳĳD�D�D�D�D�D�D�D�D�D�D~D�D�D�D�D�D�D�D�D� H + : u V M X + V " = g L ; X U @ > \ X [  ) q E ) - h I 5  B = ; C  l 2 w P W M 8 9 \ ' A F 6 U ] 6 D P I G J @ K } ;    /  �  �  f  �  5  U  !  �  i  e  �    ?  �  �  �  �  �  *  =  F  �  �  Z  m  �  8  J  {    r  �    �  2  n  �  �  �    �  �  �  ^  �  �  0  �  �  F  �  `  �    M    y  �  �  �  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  ^  X  R  L  E  >  7  .  #      �  �  �  �  �  �  �  �  �  �  �  ~  |  y  w  u  q  l  g  b  ]  X  R  J  B  9  1  )  !  !    �  �  �  �  �  �  z  _  @    �  �  �  �  m  S  2  	  C  9  /  &        �  �  �  �  �  �  �  �  z  h  V  D  2  C  �  �  �  �  �  �  �  �  ~  �  u  J    �  �  e    �  ?  �  �  �  �  �  �  �  �  w  n  �  �  �  �  }  Z  >  1  2  �      �  �  �  �  �  �  �  y  `  F  !  �  7  �  �  �  H    �  �  �  �  �  �  ~  d  F  $  �  �  �  6  �  �    �    �  M  /  %  "    	  �  �  �  W    �  |  -  �  �  \  ;  )    S  M  N  @  ;  /      �  �  �  M  
  �  m    �  T  �  �  1  &      	  �  �  �  �  �  �  v  W  7    �  �  �  o  =  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  [  I  6  �  �  �  �  �  �  �  q  [  ?    �  �    6  �  �    �  x  	x  	�  

  
  
  
  	�  	�  	�  	^  	  �  >  �    u  �  H  E  (  1  �  �  j  1  �  �  �  �  `  8      �  �  �  m    z   �  �    �  �  �  �  �  �  �  �  k  ?    �  q  "  �  :  �  ?  �  �  �  �  �  �  �  �  s  `  M  :  &    �  �  C  �  �  G  �  �  �  �  �  �    l  Y  F  1      �  �  �  c  -     �  �  �  �         	  �  �  �  �  �  �  �  h  -  �  Y  �  
  G  )    �  �  �  W  )  �  �  �  f  6  �  �  v    �  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  3  9  7  +    �  �  �  ~  F    �  �  b  )  �  �  �  h  2  �  �  �  z  w  v  u  e  O  8  $      �  �  �  �  }  J    �  �  �  �  w  �  �  �  �  �  ~  L  <  �  �  M  �  S  �  �                  �  �  �  �  �  �  �  �  �  �  i  =  �  �  �  �  �  �  �  �  �  a  3  �  �  �  ;  �  �  2  �  �  4  !       �  �  �  �  �    C  �  �  *  �  N  �  W  �  �  .  v  �  �  �  �  �  s    �  G    \  R  8    �  �  �  �  �  �  8  F  5       2  &      �  �  �  �  !  �  �  2  &  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �          �  �  �  �  �  V    �  �  E  �  C  �  
  W  O  G  ?  8  /  $        �  �  �  �  �  �  �  �  {  j  ,    0  (        �  �  �  �  �  �  ~  \  3  �  �  �  �  c  `  [  N  ?  0    
  �  �  �  �  �  y  ^  B    �  �  }  q  h  _  V  M  D  ;  4  -  '  !      	   �   �   �   �   �   {  �  �  �  �  �  L  	  �  �  >    �  �  D  �  �    �  �  �  6    �  �  �  �  �  v  P  '  �  �  �  n  A    �  �  �  �  �  �  �  v  f  Q  <  %    �  �  �  �  �  t  \  G  3  '    "            �  �  �  �  �  �  �  �  �  �  }  g  P  :  �  �  �  �  �  �    k  S  8    �  �  �  Z    �  �  ^  �     �  �  �  �  �  u  I    �  z  *  �  �  ^    �  '  S    �  �  �  �  �  �  �  �  �  z  f  L  )     �  �  M    �  t  �  �  �  �  �  �  v  L    �  �  �  x  L    �  r  �  A  A  �  �  �  �  �  �  �  �  �  �  j  D    �  �  :  �  �  &   �  �  �  �  �  �  �  �  �  �  s  i  c  ]  W  Q  J  D  >  8  2  *        �  �  �  �  �  c  8    �  �  �  h    �  `  �  ]  T  L  C  9  -  !      �  �  �  �  �  �  �  w  T  2    �  �  �  �  �  p  g  ]  T  K  A  6  *         �   �   �   �  ;  B  >  ;  C  G  _  w  v  U    �  �  3  �  v  �    �  �          �  �  �  �  �  �  �  f  D  "    �  �  �  �  �  <  w  �  �  |  s  i  ]  J  1    �  �  �  =  �  �  &  �   �  �  �  �  x  o  b  T  D  3  "    �  �  �  �  |  M    �  �  �  u  i  c  i  l  ]  :  �  �  �  F  
  �  �  J    �  {  4  �  �  �  �  �  r  J    �  �  �  _  +  �  �  Q  �  ^  �  .  +             
      �  �  �  �  �  �  �  g  I  )    h  d  `  \  W  Q  K  D  :  (      �  �  �  �  �  �  �  ~  �  �  �  �  �  {  n  `  R  C  3  #    �  �  �  �  �  �  �  �  �  z  p  e  [  R  J  ;  "    '  1  9  O  k    �  '  �    �  �  w  0  �  �  s  C    �  �  �  �  �  ]  1    �  �    p  a  S  C  4  %      �  �  �  �  �  �  |  [  +  �  �    �  ;  
�  
�  
:  	�  	�  	5  �      �  7  �  H  �  M  �  B