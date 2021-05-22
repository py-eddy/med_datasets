CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�/��v�     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�   max       P#�     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       <��
     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?h�\)   max       @F��z�H     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @v��z�H     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�2        max       @�`         ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �%   max       <�C�     ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��=   max       B4]�     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�h�   max       B4z�     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?v�   max       C���     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?PŞ   max       C���     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          :     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�   max       P��     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?� ѷX�   max       ?���>BZ�     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       <��
     ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?h�\)   max       @F��z�H     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�z    max       @v��z�H     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�2        max       @�          ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?���>BZ�     p  a�                              !   
   
                  	                           %                     1                     9               (                        
      
      
               )            	   	   /               /   NJ��N2xLO��N\�O��N �NO��O�7!N]� O0��N� �N��O4�N:�O��1Old�N{�[O�
�M�O/ߝNȯ OA\O��VO�$FO@��O�O�	�O1��N��O?�-OG��N�$&N>O�O�S�NٽO�[OV�cN�rOR�NP#�O�>NN�gSNrN.$}O��MO�OI��O�PO>�UP 3�NU��O�#N��O��OkN3N�aN�*-N_\O/��O�lP%O-{�N��WN]y�N�:�NW��Ol8lO:b�NT�uN��zM��tO�$^N3�~<��
;�`B;ě�;D��;o�D���D���t��t��49X�D���D���T���e`B�u�u�u��C���C����㼛�㼣�
���
��1��9X��9X��9X��9X��9X��9X��j�ě����ͼ�����`B�������������+�+�+�+�C��C��\)�\)�\)�\)�\)�t��t��t���P����w�#�
�'49X�49X�<j�D���D���P�`�T���Y��e`B�e`B�q���u�������
��1Z[ghprg[UUZZZZZZZZZZaahnz����znaaaaaaaaa��������������������()6;BIHB6))&((((((((���!)+0)����!%"������� �������������������������������)/0-)#������
������������������������������������������ 
#05851$
��
!#'%#
	������������������������������������������������ELTagou���tmaTNIIGEst~�����tpssssssssss%)/5@BB>85)
DNO[\gt|trkge[NBDDDDST[aimvwwwtpmaaTROPSy~����������������ygq�������������qg_]g`emz����}ohsmfa[ada`�������������������������������������������������������������������������������
#%('%# 
������#/<HUX_^UTNF</(%##����������������������������������������nbUI<60*((+0<IUajrtn�������������257BEMMHB?5420222222��������������������@BOhjkjlilih[O@999=@ghmst��������xtmhjhg#0<MUbnu|{nbUK<0����&%���������������������������������������������NO[ahtutkhc[WONNNNNN��������������������^afnz��������zqna^]^�����
���������� ).>EJLLB5)rt���������������tr���	

����������#IUdbpoibUI0#
��kny{����{nihkkkkkkkk��������������������W[gt|�����}tgd[XWWWW��������������������Zdgrvw����������tg[Z�������������������������������������)5@=65)��������������������hjqt������������|thhy������������������y���������������������������������
!#.%#
��������yz���������������~zy������������������������������������������������������������')-5750)����������������������������������������*)$��������#'*�������������������þZ�Q�Y�Z�f�s�y�}�s�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�U�T�H�G�D�C�H�U�V�Y�X�U�U�U�U�U�U�U�U�U�U�M�H�A�<�9�7�<�H�U�a�l�n�q�z�z�n�l�a�U�#���#�'�/�<�>�E�<�<�/�#�#�#�#�#�#�#�#����ƳƧƥƟƝƚƗƚƧƭƴ�����������������������������������������������������˾4�(�$�!���.�4�A�M�Z�f�t�v�s�f�Z�M�A�4���������������$�0�=�D�J�F�=�6�0�$����/�'�$�/�<�H�P�U�Y�U�H�<�/�/�/�/�/�/�/�/��ÿùøóõùÿ�������������������������{�t�g�b�g�t¦«¨¦�������������������������������������𽷽����������������}�����������ĽϽĽ���ŭŧŭűŹż������������źŹŭŭŭŭŭŭ���������Ŀοѿݿ�����������ѿĿ�ÓÍÇ�y�s�s�z�}ÇÓÚàåìýÿùìàÓ�T�N�I�J�T�`�m�v�m�j�`�[�T�T�T�T�T�T�T�T�5�(�%�*�5�Z�s�������������������s�g�A�5¿¸²±²¿��������¿¿¿¿¿¿¿¿¿¿�ѿǿĿÿĿǿѿٿݿ�������������ݿѿ��������������������������¿¿�������������������ƻ������������������������ƚƎ�h�O�6�����*�C�\�uƎƬƹƻƳƧƚőŇŀ�~ŇŚŠŭŹ��������������ŶŭŠő�"������"�0�H�T�a�m�w�m�i�T�B�;�2�"������ݿؿ׿ݿ��������������������"�/�H�T�f�k�h�a�T�H�;�/�"�������!�(�5�A�G�N�N�P�N�M�G�A�5�(����׾־Ҿʾʾʾ׾߾�������������z�v�z����������������ǾʾǾ���������5�.�*�*�(�)�6�B�L�O�[�h�t�q�h�[�U�O�B�5����������������������������������������FFFFF$F1F=F?F=F5F1F$FFFFFFFF�M�W�[�\�_�Z�N�4�(���	������4�A�M�c�W�S�W�Z�g�������������������������s�c������*�6�C�D�C�9�6�*���������������������������������������������	��������	��"�.�;�G�J�R�R�G�;�.�"��	���������������������ž׾ھ���׾ʾ����!����!�'�*�+�-�:�E�F�@�B�F�V�R�F�:�!�Y�f�w��������#�-�.�!���㼱�����f�Y�T�;�.�"������.�;�G�J�Z�g�o�q�m�`�T�h�f�b�e�h�t�uāčĖėčċčĎčā�t�h�hÓÒÎÓÛÞàâìñìèàÖÓÓÓÓÓÓ������������������������������������������߹ܹٹܹ�����3�@�U�N�D�3�.��������������������ɺкܺ���&�%����ֺ��t�i�h�d�h�n�tāčĚĦĳĵĴĭĦĚēā�t�	�	��������������������������"���	�������������������������������������������~�t�r�o�s�����������������������������N�I�A�?�A�N�Z�g�m�n�g�Z�N�N�N�N�N�N�N�N�<�4�0�(�0�<�I�U�a�n�q�{�}�{�r�n�b�U�I�<�����������	��"�(�"�����	���������;�1�0�'�����"�.�T�[�g�i�t�p�`�T�G�;���������������������)�0�-�-�)���Ƶ�������������������������������������������������������������������������������	�����������!�"�#��������������������������������������������ѻ����������������������������û����������лʻŻϻܻ������!���������ܻ��(�����2�A�Z�k�s�����������{�g�N�A�(�L�C�B�J�L�Y�`�e�r�z�~�����~�|�r�n�e�Y�L������������������������F1F=FJFTFVFcFgFcFVFJF=F1F1F1F1F1F1F1F1F1���������������������ʾ׾ھ׾Ѿʾ������������������ûƻлܻ�ܻлû�������������������þ��������������#�!����������#��
� ���������
��#�.�5�<�H�M�I�<�/�#�������������������������������������������������������������������������������������������������������������������������"�.�:�D�P�W�G�8�!���������������"��
����'�4�@�D�@�4�+�'������� > � 9 7 � X ) D \ 1 t S P  B N y u B & a K w 5  # . 7 ` % ; Z c   [ _ 6 8 C � j ] < v M Z J M ^  9 D K U J Y n 3 h p . t 3 @ 8 O F i H K A P l  {    Z  �  9  t  �  M    [  y  w  �  �  �  �  $    �      u    Z    �    8  �  y  �  �  �  �  v  /  �  �  I  �    "  �  o    |  ^  s  �  �  �  �  a  a  ~    �  j    �  0  ]  |  G  i  |  �  �    �     �  r  �  #    �<�C�;o�t��D���o�o��C��o�D���49X��1��9X��j��t��49X�C���t����ͼ����w���+�#�
�L�ͼ��\)�m�h�\)��/�<j�@�������P�����<j�t��0 Ž0 Ž0 Ž8Q콴9X�L�ͽ8Q�t���㽕���o�L�ͽ@��]/��%��P�@��8Q�P�`�<j�49X�D���L�ͽ49X��7L�u�� Ž�t��]/����y�#�y�#���`���w��+��hs��%��Q�B		�B�B!�B��B��B�B��BϮB��B��BƄBPkB$��B'BGfB��B-��A�&UB�oBz�BªA��=B ��B
�QA���BS�B�B�B"��B$s�B��B!&IB�B&�uBԅB��B�wBB�IB&gDB-�B�~B[�BtuB�B7�B#�B�B�B�B%��B(�0BK�B	��BL�B	�nBՃB4]�B^B�B�BT�B��B!��B$��BeB.�B3�B�lB�	B�xBg�Ba(B��B��B�6B}�B!B�iBA�B@B?�B�B]{B�&BS�B@B%*�B=�B@:Bp�B-�RA�h�B�mB}VB�IA�uBJ�B&�A���B'tB��B=�B"�,B$~�B�#B!>}B��B&�B�(B�|BëB�2BE�B&3B-�RB@YB��B{BK�B?�B"��B�DB@jB�B%;B(�(Bi=B	�;B:�B
{B4�B4z�B��B��B��B;�BV�B!�B$@?B@�B;�B9�B�B��B��B��B-^B̉B��AAr'A�R�Aż4A�B��A�c�A<QB	��Aú�A�[BA�9oA�ɦA"Z�A��hA}��A�S�Ah�A���A�ƂA}ĥAtE�B1LB��A�@(A�,uA�|kA���A�V�ATa#AI�&A�%HAK�C�̹A8-�A���A���A�� A_{�AOܵ@r��A �Acy�A�.A�@�'�?v�@DH�Aރ�A� �A��A�V�A�ExA��A[l�Ad�A�QB�1AK@�A���A�[�@��@�W�A�R?��O@��PC���AMv�@�2�A�&MA�QA��*A���A��{A��@ɍ�AA�AĢ�A�A�qVB�A�p�A; B	�EA��A�p�A��nA�R�A �yA��A}=�A�U9AgQA��MA��A}3WAu �BisB-�A���A���A�mWA��A��AU2AIP�A�t�AI6C�ƝA8�A�ݨA��aA��A`��AO#@t'�A֌Aa<A� �A���@�=.?PŞ@<�A�b7A���A��A�A�~DA�x�A[Ab�A�d�B�WAK�(A���A��@��C@��cA���?��@�5C���AMP@���A�y�A�{mA���A�}�A��A�-@��                              "   
                     	                     	      %                     2                     :               (                        
      
      
               )            
   	   /               /                                                         %               +   !                                                   9                  %            '                                    )                                                                                          %               +                                                      +                  #            #                                    %                                    NJ��N2xLO��N\�O��N �NOo�uO(� N]� O'�LN� �N+�O4�N:�O��Old�N{�[O�
�M�N���Nm�kOA\O��VO���OH�O�Ox�oO1��N��N�ٲO*TN�$&N>O0�O���NL�N`��OV�cN�rNɤ�P��O�>NN�gSNrN.$}O�!�O�bOI��O�PN�<SO�W.NU��O�N��O��OkN3N�aN�*-N_\O/��O�lO���O�N��WN]y�N�:�NW��O�QO:b�NT�uN��zM��tO�$^N3�~  �  �  �  )  /  Z  H  �  �  �  X  �  Z  ?  �  k  ?  �  @  �  `  M    �  ^  �  �      �  
  %  /    O  "  �  ,  N  #    K  �  _  �    5  �  �  O  Y  +  d  9  �  �  8  &  �  �  �  �  W    k    �  (  
I  �  �  �  @    �<��
;�`B;ě�;D��;o�D����o��o�t��D���D���u�T���e`B��j�u�u��C���C���j��9X���
���
��/��j��9X�o��9X��9X����`B�ě����ͽ'�h���+�����\)��P�+�+�+�+�t��\)�\)�\)�#�
��P�\)��P�t��t���P����w�#�
�'49X�49X�D���L�ͽD���P�`�T���Y���+�e`B�q���u�������
��1Z[ghprg[UUZZZZZZZZZZaahnz����znaaaaaaaaa��������������������()6;BIHB6))&((((((((���!)+0)����!%"���������� �����������������������������)/0-)#�����

��������������������������������
����������� 
#05851$
��
!#'%#
	������������������������������������������������ELTagou���tmaTNIIGEst~�����tpssssssssss	)57<951)
				INT[fgkngfa[NJIIIIIIST[aimvwwwtpmaaTROPSy~����������������ygmt��������������wggahmz�����{mda_]aebaa��������������������������������������������������������������������������������
#$#"
���!#(/<HSSOHHA</-)%#!!����������������������������������������/0:<IU`fiib]UI<;40./����  
��������15BDKKCBA55311111111��������������������@BOhjkjlilih[O@999=@ghmst��������xtmhjhg"#0<FIUbcbWUI<0'#"""�����# �������������������������������������������NO[ahtutkhc[WONNNNNN��������������������_ajnz��������zsna_^_�����
���������� ).>EJLLB5)rt���������������tr������


����������#Ubkif`RI0#
��kny{����{nihkkkkkkkk��������������������W[gt|�����}tgd[XWWWW��������������������Zdgrvw����������tg[Z�������������������������������������)5@=65)��������������������hjqt������������|thhy������������������y����������������������������������
!#.%#
��������yz���������������~zy������������������������������������������������������������')-5750)����������������������������������������*)$��������#'*�������������������þZ�Q�Y�Z�f�s�y�}�s�f�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�U�T�H�G�D�C�H�U�V�Y�X�U�U�U�U�U�U�U�U�U�U�M�H�A�<�9�7�<�H�U�a�l�n�q�z�z�n�l�a�U�#���#�'�/�<�>�E�<�<�/�#�#�#�#�#�#�#�#����ƳƧƥƟƝƚƗƚƧƭƴ�����������������������������������������������������˾M�A�4�(�&�$� �$�2�4�A�M�[�f�r�u�s�f�Z�M�������������$�0�=�D�A�=�6�0�'�$��/�'�$�/�<�H�P�U�Y�U�H�<�/�/�/�/�/�/�/�/����ùóöù�����������������������������{�t�g�b�g�t¦«¨¦���������������������������������������𽷽����������������}�����������ĽϽĽ���ŭŧŭűŹż������������źŹŭŭŭŭŭŭ�Ŀ����Ŀѿܿݿ�������������ݿѿ�ÓÍÇ�y�s�s�z�}ÇÓÚàåìýÿùìàÓ�T�N�I�J�T�`�m�v�m�j�`�[�T�T�T�T�T�T�T�T�5�(�%�*�5�Z�s�������������������s�g�A�5¿¸²±²¿��������¿¿¿¿¿¿¿¿¿¿�ݿֿѿʿǿϿѿݿ�������������ݿݿݿݿ�������������������������������������������������ƻ������������������������ƚƎ�h�O�6�����*�C�\�uƎƬƹƻƳƧƚŠŔōŇŇŒşŠŭŹ����������������ŭŠ�!��
���"�)�/�;�H�T�a�e�a�T�E�;�/�(�!������ݿؿ׿ݿ��������������"� ���"�/�5�;�H�T�_�d�a�[�T�J�H�;�/�"������!�(�5�A�G�N�N�P�N�M�G�A�5�(����׾־Ҿʾʾʾ׾߾�����������㾘������}������������������������������B�?�6�1�-�.�6�B�O�Z�[�h�n�k�h�[�O�H�B�B����������������������������������������FFFFF$F1F=F?F=F5F1F$FFFFFFFF��������(�4�A�I�M�Q�S�P�M�A�4�(��e�X�T�Z�g�s�������������������������s�e�����*�6�C�C�C�7�6�*������������������������������������������������	��������	��"�.�;�G�J�R�R�G�;�.�"��	���������������������ž׾ھ���׾ʾ����!����!�$�+�-�-�.�:�:�@�F�E�:�0�-�!�!��|�~�����������!�&�&�� ��ʼ�����T�;�.�"������.�;�G�J�Z�g�o�q�m�`�T�h�f�b�e�h�t�uāčĖėčċčĎčā�t�h�hÓÒÎÓÛÞàâìñìèàÖÓÓÓÓÓÓ�������������������������������������������ܹڹ޹�����'�3�C�K�A�3�,������ú������������ɺκۺ���$�$����ֺ��t�i�h�d�h�n�tāčĚĦĳĵĴĭĦĚēā�t�	�	��������������������������"���	��������������������������������������������������v�v���������������������������N�I�A�?�A�N�Z�g�m�n�g�Z�N�N�N�N�N�N�N�N�<�6�0�0�<�I�J�U�b�n�z�{�|�{�p�n�b�U�I�<�����������	��"�(�"�����	���������;�1�0�'�����"�.�T�[�g�i�t�p�`�T�G�;���������������������)�0�-�-�)���Ƶ�������������������������������������������������������������������������������	�����������!�"�#��������������������������������������������ѻ����������������������������û����������лʻŻϻܻ������!���������ܻ������+�4�A�g�u�}�������y�g�N�A�5�(��L�F�D�D�L�L�Y�c�e�r�u�~�����~�z�r�i�Y�L������������������������F1F=FJFTFVFcFgFcFVFJF=F1F1F1F1F1F1F1F1F1���������������������ʾ׾ھ׾Ѿʾ������������������ûƻлܻ�ܻлû��������������������������������������������
������#��
� ���������
��#�.�5�<�H�M�I�<�/�#�������������������������������������������������������������������������������������������������������������������������"�.�:�D�P�W�G�8�!���������������"��
����'�4�@�D�@�4�+�'������� > � 9 7 � X + 9 \ / t M P  6 N y u B  ^ K w 5 u # ! 7 ` ) - Z c  a U b 8 C b Z ] < v M U K M ^  6 D L U J Y n 3 h p . t 1 = 8 O F i 8 K A P l  {=�  Z  �  9  t  �  M  �  g  y  o  �  \  �  �  T    �        �  Z    L  �  8  �  y  �  �  %  �  v  {  M  u  �  �    @  �  o    |  ^  5  �  �  �  
  %  a  M    �  j    �  0  ]  |  G    U  �  �    �  Y  �  r  �  #    �  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �                             
      $  �  �  ~  p  a  R  @  +    �  �  �  �  �  S    �  �  [    )  )  *  *  +  +  +  *  &  #      �  �  �  �  �  �  }  d  /  '          �  �  �  �  �  �  �  �  �  �  v  f  V  G  Z  M  A  4  &      �  �  �  �  �  �  b  ?    �  �  �  k  G  H  F  A  5  &      �  �  �  �  �  �  �  �  �  �  ^  &  b  y  �  �  �  �  �  �  �  u  \  5  �  �  a    �  P  �  b  �  �  �  �  �  �  �  �  �  �  �  �        )  6  D  Q  _  �  �  �  �  �  �  �  �  �  z  X  2    �  �  A  �  @  v  =  X  Q  I  A  9  6  5  1  +  %  !        �  �  �  �  h  �  �  �  �  �  �  �  �  �  �  {  o  c  U  F  1    �  �  i  �  Z  T  M  >  0      �  �  �  �  �  �  u  X  9    �  �  q  ?  +      �  �  �  �  �  �  �  �  `  ?    �  �  �  �  {  �    @  [  p  }  z  k  V  ?  !  �  �  �  d    i  �      k  ]  N  6    �  �  y  `  5    �  �  `  (  �  �  y    �  ?  8  2  ,  &              �   �   �   �   �   �   �   �   �   �  �  �  �  �    i  M  1    �  �  �  �  �  �  �  �  �  w  m  @  <  8  4  0  ,  (  $                           �  �  �  �  �  �  �  �  �  v  e  O  6    �  �  �  h  +  �  <  C  J  Q  X  _  V  J  9  %    �  �  �  B    �  �  ;   �  M  E  ;  /      �  �  �  �  |  L    �  �  Q    �  N   �      �  �  �  �  �  �  c  0  �  �  �  {  e  Z  H  )      �  �  �  �  �  �  �  �  �  �  _  7    �  �  M  �  }    x  E  P  Z  Y  Q  H  A  ;  )      �  �  �  �  �  �  �  f  M  �  �  �  �  �  �  �  �  p  _  N  6    �  �  �  �  W  *  �  W  �  �  �  �  �  �  �  �  �  k  ?    �  �  f  �  +  u  �        �  �  �  �  �  �  �  w  d  O  3    �  �  H     �           �  �  �  �  �  �  �  �  �  �  �  �  �  �  b  ?    8  _  ~  �  �  �  �  �  �  t  \  @  "  �  �  �  e  ,  �  �  �  �    
  	      �  �  �  �  {  0  �  �  !  �     �  %  !                   �   �   �   �   �   �   �   �   �   �  /      �  �  �  �  �  l  J  $  �  �  �  �  a    �  �  v  �  �  �            �  �  �  i  ,  �  �     z  �  �  �  N  O  I  ;  -    	  �  �  �  �  �  r  5  �  �  #  �  u  u  	      !         �  �  �  �  s  D    �  �  �  h  =    x  �  �  �  �  �  #  A  S  Y  K  0    �  �  �  O    �  �  ,  #            �  �  �  �  x  S  +  �  �  �  w  A    N  K  H  F  C  >  6  -  $        �  �  �  �  �  g  :                !        �  �  �  �  �  c  5  �  l    �  W    |  r  j  _  J     �  �  o  !  �  P  �  ~  
  o  �  K  F  B  =  2  #    �  �  �  �  `  3    �  y  )  �  �  *  �  �  �  �  w  f  [  L  :  $    �  �  �  �  �  i  L  >  :  _  ]  Z  X  U  S  Q  P  R  S  T  U  V  X  [  ]  `  c  f  i  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    <  �      �  �  o  6  �  �  �  l  /  �  �  B  �  +  n  �  �  )  3  *      �  �  �  �  �  f  5  �  �  �  s  L    �  t  �  �  �  w  i  X  F  3    	  �  �  �  �  j  H    �  �  �  �  �  �  �  �  �  �  m  S  :  #    �  �  �  �  K  (    �    "  4  B  L  O  L  F  =  .    �  �  �  �  c  5  �  z  
  =  T  V  I  8  '    �  �  �  �  w  P  $  �  �  4  �  Q   �  +  "         �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  a  c  c  ]  U  I  <  .  !       �  �  �  t  U  9  !  >  n  9  /  %        �  �  �  �  �  �  �  g  @    �  �  �  w  �  �  w  m  `  U  F  5      �  �  �  ]  2    �  �  �  Y  �  �  �  �  �  w  \  B  '    �  �  �  �        �  �  �  8  '      �  �  �  �  �  �  �  �  �  �  �  �        �  &    
    �  �  �  �  �  �  �  �  �  �  q  ]  H  2      �  �  �  �  �  �  �  �  r  \  B  !  �  �  �  u    �  v  4  �    t  j  `  U  K  ?  3  &        �  �  �  �  �  �  �  �  �  �  �  �  �  p  J    �  �  �  i  9  �  �  W  *  9  _  �  �  �  �  x  v  ^  K  J  k  �  �  �  �  �  �  �  �  �  �  M  V  G  C  C  F  H  C  :  $    �  �  z  >     �  �  a  \             �  �  �  �  l  C    �  �  �  %  �  I  �  �  k  [  L  =  $    �  �  �            �  �  �  �  �  �    �  �  �    i  R  9    �  �  �  �  U  (  �  �  �  e  2  �  �  p  `  P  C  8  1  -  #       �  �  �  t  F     �   �  (    �  �  �  �  �  x  `  H  1      �  �  �  �  �  �  �  	�  
  
4  
@  
I  
G  
6  
  	�  	�  	,  �  ^  �  *  �    M  ]  c  �  �  �  �  �  �  l  A      �  �  �  �  '  �  .  �     |  �  �  �  �  �  �  �  �  �  �  �  �  o  R  2    �  �  �  x  �  �  �  �  �  }  n  \  6      �  �  A  �  �  7  �  T  �  @  6  ,  "        �  �  �  �  �  �  �  �  �  v  ^  G  0    �  �  �  �  �  �  Y  +  �  �  y  7  �  �  E  �  �  �  z  �  m  U  >  $  
  �  �  �  �  k  4  �  �  �  q  E    �  �